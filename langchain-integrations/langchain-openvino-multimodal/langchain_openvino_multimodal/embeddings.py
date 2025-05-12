from pathlib import Path
from typing import Any, Dict, List
import openvino as ov
import numpy as np
import requests

from langchain_core.embeddings import Embeddings
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image

DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_EN = (
    "Represent this question for searching relevant passages: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："

'''
Source: langchain_community.embeddings.openvino
'''
class OpenVINOEmbeddings(BaseModel, Embeddings):
    """OpenVINO embedding models.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import OpenVINOEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'CPU'}
            encode_kwargs = {'normalize_embeddings': True}
            ov = OpenVINOEmbeddings(
                model_name_or_path=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    ov_model: Any = None
    """OpenVINO model object."""
    tokenizer: Any = None
    """Tokenizer for embedding model."""
    model_name_or_path: str
    """HuggingFace model id."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    show_progress: bool = False
    """Whether to show a progress bar."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        try:
            from optimum.intel.openvino import OVModelForFeatureExtraction
        except ImportError as e:
            raise ImportError(
                "Could not import optimum-intel python package. "
                "Please install it with: "
                "pip install -U 'optimum[openvino,nncf]'"
            ) from e

        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise ImportError(
                "Could not import huggingface_hub python package. "
                "Please install it with: "
                "`pip install -U huggingface_hub`."
            ) from e

        def require_model_export(
            model_id: str, revision: Any = None, subfolder: Any = None
        ) -> bool:
            model_dir = Path(model_id)
            if subfolder is not None:
                model_dir = model_dir / subfolder
            if model_dir.is_dir():
                return (
                    not (model_dir / "openvino_model.xml").exists()
                    or not (model_dir / "openvino_model.bin").exists()
                )
            hf_api = HfApi()
            try:
                model_info = hf_api.model_info(model_id, revision=revision or "main")
                normalized_subfolder = (
                    None if subfolder is None else Path(subfolder).as_posix()
                )
                model_files = [
                    file.rfilename
                    for file in model_info.siblings
                    if normalized_subfolder is None
                    or file.rfilename.startswith(normalized_subfolder)
                ]
                ov_model_path = (
                    "openvino_model.xml"
                    if subfolder is None
                    else f"{normalized_subfolder}/openvino_model.xml"
                )
                return (
                    ov_model_path not in model_files
                    or ov_model_path.replace(".xml", ".bin") not in model_files
                )
            except Exception:
                return True

        if require_model_export(self.model_name_or_path):
            # use remote model
            self.ov_model = OVModelForFeatureExtraction.from_pretrained(
                self.model_name_or_path, export=True, **self.model_kwargs
            )
        else:
            # use local model
            self.ov_model = OVModelForFeatureExtraction.from_pretrained(
                self.model_name_or_path, **self.model_kwargs
            )

        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Unable to import transformers, please install with "
                "`pip install -U transformers`."
            ) from e
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def _text_length(self, text: Any) -> int:
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        # Empty string or list of ints
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            # Sum of length of individual strings
            return sum([len(t) for t in text])

    def encode(
        self,
        sentences: Any,
        batch_size: int = 4,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        mean_pooling: bool = False,
        normalize_embeddings: bool = True,
    ) -> Any:
        """
        Computes sentence embeddings.

        :param sentences: the sentences to embed.
        :param batch_size: the batch size used for the computation.
        :param show_progress_bar: Whether to output a progress bar.
        :param convert_to_numpy: Whether the output should be a list of numpy vectors.
        :param convert_to_tensor: Whether the output should be one large tensor.
        :param mean_pooling: Whether to pool returned vectors.
        :param normalize_embeddings: Whether to normalize returned vectors.

        :return: By default, a 2d numpy array with shape [num_inputs, output_dimension].
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "Unable to import numpy, please install with `pip install -U numpy`."
            ) from e
        try:
            from tqdm import trange
        except ImportError as e:
            raise ImportError(
                "Unable to import tqdm, please install with `pip install -U tqdm`."
            ) from e
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "Unable to import torch, please install with `pip install -U torch`."
            ) from e

        def run_mean_pooling(model_output: Any, attention_mask: Any) -> Any:
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings: Any = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]

            length = self.ov_model.request.inputs[0].get_partial_shape()[1]
            if length.is_dynamic:
                features = self.tokenizer(
                    sentences_batch, padding=True, truncation=True, return_tensors="pt"
                )
            else:
                features = self.tokenizer(
                    sentences_batch,
                    padding="max_length",
                    max_length=length.get_length(),
                    truncation=True,
                    return_tensors="pt",
                )

            out_features = self.ov_model(**features)
            if mean_pooling:
                embeddings = run_mean_pooling(out_features, features["attention_mask"])
            else:
                embeddings = out_features[0][:, 0]
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
            if convert_to_numpy:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            if len(all_embeddings):
                all_embeddings = torch.stack(all_embeddings)
            else:
                all_embeddings = torch.Tensor()
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.encode(
            texts, show_progress_bar=self.show_progress, **self.encode_kwargs
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    def save_model(
        self,
        model_path: str,
    ) -> bool:
        self.ov_model.half()
        self.ov_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        return True


class OpenVINOBgeEmbeddings(OpenVINOEmbeddings):
    """OpenVNO BGE embedding models.

    Bge Example:
        .. code-block:: python

            from langchain_community.embeddings import OpenVINOBgeEmbeddings

            model_name = "BAAI/bge-large-en-v1.5"
            model_kwargs = {'device': 'CPU'}
            encode_kwargs = {'normalize_embeddings': True}
            ov = OpenVINOBgeEmbeddings(
                model_name_or_path=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    query_instruction: str = DEFAULT_QUERY_BGE_INSTRUCTION_EN
    """Instruction to use for embedding query."""
    embed_instruction: str = ""
    """Instruction to use for embedding document."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        if "-zh" in self.model_name_or_path:
            self.query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_ZH

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [self.embed_instruction + t.replace("\n", " ") for t in texts]
        embeddings = self.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.encode(self.query_instruction + text, **self.encode_kwargs)
        return embedding.tolist()

''' End of langchain_community.embeddings.openvino'''

class OpenVINOClipEmbeddings(Embeddings):
    """OpenvinoClip embedding model integration.

    Setup:
        Install ``langchain-openvino-clip``

        .. code-block:: bash

            pip install -U langchain-openvino-clip

    Key init args — completion params:
        model: str
            Name of OpenvinoClip model to use.
        device: str
            Device to use for inference. Options: "GPU", "CPU", "NPU".
        ov_model_path: str
            Path to OpenVINO model file. 

    See full list of supported init args and their descriptions in the params section.
    NPU does not support text embeddings currently due to static shape limitation.

    Instantiate:
        .. code-block:: python

            from langchain_openvino_clip import OpenvinoClipEmbeddings

            embed = OpenvinoClipEmbeddings(
                model_id="openai/clip-vit-base-patch32",
                device="GPU",
            )

    Embed single text:
        .. code-block:: python

            input_text = "A photo of a cat"
            embed.embed_query(input_text)

    Embed multiple text:
        .. code-block:: python

            input_texts = ["Document 1...", "Document 2..."]
            embed.embed_documents(input_texts)

        .. code-block:: python

    Embed single image:
        .. code-block:: python

            input_image = "path/to/image.jpg"
            embed.embed_image(input_image)
    
    Embed multiple images:
        .. code-block:: python

            input_images = ["path/to/image1.jpg", "path/to/image2.jpg"]
            embed.embed_images(input_images)

    """

    model_id: str = "openai/clip-vit-base-patch32"
    device: str = "GPU"
    
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32",
                 device: str = "GPU",
                 ov_model_path: str = "clip-vit-base-patch32-fp16.xml") -> None:
        """Initialize OpenVINO CLIP model."""
        
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.model.eval()
        
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        text = ["a photo of a cat"]
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
                
        ov_model_path = Path(ov_model_path)
        if not ov_model_path.exists():
            print(f"Converting model to OpenVINO format and saving to {ov_model_path}")
            
            if device == "NPU":
                input_shapes = {
                    "input_ids": inputs["input_ids"].shape,
                    "pixel_values": (1, 3, 224, 224),
                    "attention_mask": inputs["input_ids"].shape
                }
            
                self.model.config.torchscript = True
                ov_model = ov.convert_model(self.model, input=input_shapes, example_input=dict(inputs))

                correct_names = ["input_ids", "pixel_values", "attention_mask"]
                for i, ov_input in enumerate(ov_model.inputs):
                    ov_input.set_names({correct_names[i]})
            
            else:
                input_shapes = {
                    "input_ids": (-1, -1),
                    "pixel_values": (1, 3, 224, 224),
                    "attention_mask": (-1, -1)
                }
            
                self.model.config.torchscript = True
                ov_model = ov.convert_model(self.model, input=input_shapes, example_input=dict(inputs))

                correct_input_names = ["input_ids", "pixel_values", "attention_mask"]

                for i, ov_input in enumerate(ov_model.inputs):
                    ov_input.set_names({correct_input_names[i]})
                    
                for i, ov_output in enumerate(ov_model.outputs):
                    if i == 2:
                        ov_output.set_names({"text_embeds"})
                        
            ov.save_model(ov_model, ov_model_path)
        
        core = ov.Core()
        self.ov_clip_model = core.compile_model(ov_model_path, device)
        print(f"Model {ov_model_path} loaded successfully on {device} device.")
        

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        text_embeddings = []
        for text in texts:
            text_embedding = self.embed_query(text)
            text_embeddings.append(text_embedding)
        
        return text_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        if self.device == "NPU":
            raise ValueError("NPU device is not supported for text embedding.")
        
        if text:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = dict(inputs)
            text_embedding = self.ov_clip_model(inputs)["text_embeds"][0]
            text_embedding = text_embedding / np.linalg.norm(text_embedding)
            return text_embedding
        else:
            print("Text is empty.")
            return None
        
    def embed_images(self, image_uris: List[str]) -> List[List[float]]:
        """Embed images."""
        image_embeddings = []
        for image in image_uris:
            image_embedding = self.embed_image(image)
            image_embedding = image_embedding / np.linalg.norm(image_embedding)
            image_embeddings.append(image_embedding)
        
        return image_embeddings
        
    def embed_image(self, image: str | np.ndarray) -> List[float]:
        """Embed image."""
        if isinstance(image, str):
            image = Path(image)
            if not image.exists():
                raise ValueError(f"Image file {image} does not exist.")
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            raise ValueError("Image must be a file path or a numpy array.")
        
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        inputs = dict(inputs)
        image_embedding = self.ov_clip_model(inputs)["image_embeds"][0]
        
        return image_embedding
import requests
import openvino as ov
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

from typing import List

from langchain_core.embeddings import Embeddings
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer


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
                ov_model = ov.convert_model(self.model, inputs=input_shapes, example_input=dict(inputs))

                correct_names = ["input_ids", "pixel_values", "attention_mask"]
                for i, ov_input in enumerate(ov_model.inputs):
                    ov_input.set_names({correct_names[i]})
            
            else:
                self.model.config.torchscript = True
                ov_model = ov.convert_model(self.model, example_input=dict(inputs))

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
            return text_embedding
        else:
            print("Text is empty.")
            return None
        
    def embed_images(self, image_uris: List[str]) -> List[List[float]]:
        """Embed images."""
        image_embeddings = []
        for image_uri in image_uris:
            image_embedding = self.embed_image(image_uri)
            image_embeddings.append(image_embedding)
        
        return image_embeddings
        
    def embed_image(self, image_uri: str) -> List[float]:
        """Embed image."""
        image_uri = Path(image_uri)
        if not image_uri.exists():
            raise ValueError(f"Image file {image_uri} does not exist.")
        
        image = Image.open(image_uri)
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        inputs = dict(inputs)
        
        image_embedding = self.ov_clip_model(inputs)["image_embeds"][0]
        
        return image_embedding
    

    


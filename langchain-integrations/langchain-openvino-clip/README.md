# langchain-openvino-clip

This package contains the LangChain integration with OpenVINOClip

## Installation

```bash
pip install -U langchain-openvino-clip
```

## Embeddings

`OpenVINOClipEmbeddings` class exposes embeddings from OpenVINOClip.

```python
from langchain_openvino_clip import OpenVINOClipEmbeddings

# Default model is: "openai/clip-vit-base-patch32" and Default device is GPU.
# Possible device values for Image embeddings are "CPU, GPU, NPU".
# Possible device values for Text embeddings are "CPU, GPU". NPU is not supported.
embeddings = OpenvinoClipEmbeddings(
                model_id="openai/clip-vit-base-patch32",
                device="GPU",
            )

# Embed single text:
input_text = "A photo of a cat"
embed.embed_query(input_text)

# Embed multiple text:
input_texts = ["text 1...", "text 2..."]
embed.embed_documents(input_texts)

# Embed single image:
input_image = "path/to/image.jpg"
embed.embed_image(input_image)

# Embed multiple images:
input_images = ["path/to/image1.jpg", "path/to/image2.jpg"]
embed.embed_images(input_images)
```

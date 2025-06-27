# langchain-openvino-multimodal

This package contains the LangChain integration with OpenVINOBlip, OpenVINOClip, and OpenVINOBgeEmbeddings

## Installation

```bash
pip install -U langchain-openvino-multimodal
```

## Embeddings

This class includes the two OpenVINO classes from `langchain_community.embeddings.openvino`. These classes were inlcuded for a unified class to create all types of OpenVINO embeddings from a single langchain package. 

For sample usage, Please see - [Usage](https://python.langchain.com/docs/integrations/text_embedding/openvino/)
1. `OpenVINOEmbeddings`
2. `OpenVINOBgeEmbeddings`

This package includes 2 new classes for OpenVINO multimodal embeddings.

1. CLIP embeddings (images and text) called `OpenVINOClipEmbeddings` class.
2. BLIP embeddings called `OpenVINOBlipEmbeddings` class.

## OpenVINOCLIP Embeddings

Currently supported CLIP model is: `openai/clip-vit-base-patch32`

### Usage

```python
from langchain_openvino_multimodal import OpenVINOClipEmbeddings

# Default model is: "openai/clip-vit-base-patch32" and Default device is GPU.
# Possible device values for Image embeddings are "CPU, GPU, NPU".
# Possible device values for Text embeddings are "CPU, GPU". NPU is not supported.
embed = OpenvinoClipEmbeddings(
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

# Embed images input as numpy arrays:
# Multiple np arrays can also be passed via ov_clip.embed_images([list of images as numpy arrays])
image = Image.open("path/to/image")
arr = np.array(image)
# Embed input: numpy array
embed.embed_image(arr)
```

## OpenVINOBLIP Embeddings

Currently supported BLIP model is: `Salesforce/blip-itm-base-coco`

### Usage

```python
from langchain_openvino_multimodal import OpenVINOBlipEmbeddings

# Default model is: "Salesforce/blip-itm-base-coco" and Default device is GPU.
# Possible device values for Image embeddings are "CPU, GPU, NPU".
# Possible device values for Text embeddings are "CPU, GPU". NPU is not supported.

# Embeddings generation may be offloaded to different devices.
# For ex: Text embeddings could be created on GPU and Image embeddings on an NPU if available.
embeddings = OpenvinoClipEmbeddings(
                model_id="openai/clip-vit-base-patch32",
                ov_text_device="GPU",
                ov_vision_device="NPU"
            )

# Embed single text:
input_text = "A photo of a cat"
embeddings.embed_query(input_text)

# Embed multiple text:
input_texts = ["text 1...", "text 2..."]
embeddings.embed_documents(input_texts)

# Embed single image:
input_image = "path/to/image.jpg"
embeddings.embed_image(input_image)

# Embed multiple images:
input_images = ["path/to/image1.jpg", "path/to/image2.jpg"]
embeddings.embed_images(input_images)

# Embed images input as numpy arrays:
# Multiple np arrays can also be passed via ov_blip.embed_images([list of images as numpy arrays])
image = Image.open("path/to/image")
image = np.array(image)
# Embed input: numpy array
image_embedding_arr = embeddings.embed_image(image)
```
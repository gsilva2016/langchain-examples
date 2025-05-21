from importlib import metadata

from langchain_openvino_multimodal.embeddings import OpenVINOClipEmbeddings, OpenVINOBgeEmbeddings, OpenVINOEmbeddings, OpenVINOBlipEmbeddings

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "OpenVINOClipEmbeddings",
    "OpenVINOBgeEmbeddings",
    "OpenVINOEmbeddings",
    "OpenVINOBlipEmbeddings",
    "__version__",
]

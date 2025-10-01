import numpy as np
import cv2
from  common.tracker.deepsort_utils.detection import compute_color_for_labels
import openvino as ov


class ReIDExtractor:
    """
    ReIDExtractor wraps an OpenVINO model for person re-identification feature extraction.
    Used to extract embeddings for detected person crops for tracking.
    """
    def __init__(self, model_path, device="AUTO"):
        """
        Initialize the re-identification extractor with a given model path and device.
        """
        self.core = ov.Core()
        self.model = self.core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]
        input_shape = list(self.input_shape)
        input_shape[0] = -1  # dynamic batch size
        self.model.reshape({self.model.inputs[0]: input_shape})
        self.compiled_model = self.core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled_model.output(0)

    def preprocess(self, frame):
        """
        Preprocess a frame for model inference: resize, transpose, and expand dims.
        """
        resized_image = cv2.resize(frame, (self.width, self.height))
        resized_image = resized_image.transpose((2, 0, 1))
        input_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
        return input_image

    def batch_preprocess(self, img_crops):
        """
        Preprocess a batch of cropped images for re-identification model.
        """
        img_batch = np.concatenate([self.preprocess(img) for img in img_crops], axis=0)
        return img_batch

    def predict(self, img_batch):
        """
        Run inference on a batch of images and return feature embeddings.
        """
        result = self.compiled_model(img_batch)[self.output_layer]
        return result

def cosin_metric(x1, x2):
    """
    Compute cosine similarity between two feature vectors.
    """
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def draw_boxes(img, bbox, identities=None):
    """
    Draw bounding boxes and identity labels on an image for visualization.
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = f"{id}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1.6, [255, 255, 255], 2)
    return img

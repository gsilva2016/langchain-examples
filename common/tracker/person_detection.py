import openvino as ov
import numpy as np
import cv2

class PersonDetector:
    """
    PersonDetector wraps an OpenVINO model for person detection.
    It loads the model, prepares input/output layers, and provides methods for inference and result processing.
    """
    def __init__(self, model_path, device="AUTO", thresh=0.5):
        """
        Initialize the person detector with a given model path, device, and detection threshold.
        """
        self.core = ov.Core()
        self.model = self.core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]
        self.model.reshape({self.model.inputs[0]: list(self.input_shape)})
        self.compiled_model = self.core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled_model.output(0)
        self.thresh = thresh

    def preprocess(self, frame):
        """
        Preprocess a frame for model inference: resize, transpose, and expand dims.
        """
        resized_image = cv2.resize(frame, (self.width, self.height))
        resized_image = resized_image.transpose((2, 0, 1))
        input_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
        return input_image

    def predict(self, frame):
        """
        Run inference on a frame and return raw model output.
        """
        input_image = self.preprocess(frame)
        result = self.compiled_model(input_image)[self.output_layer]
        return result

    def process_results(self, h, w, results):
        """
        Post-process model output to extract bounding boxes, labels, and scores.
        Returns only detections above the detection threshold.
        """
        detections = results.reshape(-1, 7)
        boxes, labels, scores = [], [], []
        for detection in detections:
            _, label, score, xmin, ymin, xmax, ymax = detection
            if score > self.thresh:
                # Convert normalized coordinates to pixel values and box format (center x/y, width, height)
                boxes.append([
                    (xmin + xmax) / 2 * w,
                    (ymin + ymax) / 2 * h,
                    (xmax - xmin) * w,
                    (ymax - ymin) * h,
                ])
                labels.append(int(label))
                scores.append(float(score))
        if len(boxes) == 0:
            boxes = np.array([]).reshape(0, 4)
            scores = np.array([])
            labels = np.array([])
        return np.array(boxes), np.array(scores), np.array(labels)
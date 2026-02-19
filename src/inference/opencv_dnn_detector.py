import cv2
import numpy as np
from pathlib import Path
from src.utils.preprocessing import letterbox


class OpenCVDNNDetector:

    def __init__(self, model_path, input_size=640, conf_threshold=0.5, nms_threshold=0.45):

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        print(f"Loading YOLOv5 ONNX model: {model_path}")

        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        print("OpenCV DNN initialized")

    def infer(self, frame):

        orig_h, orig_w = frame.shape[:2]

        img, scale, left, top = letterbox(frame, new_shape=self.input_size)

        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=False,
            crop=False,
        )

        self.net.setInput(blob)
        outputs = self.net.forward()

        if len(outputs.shape) == 3:
            outputs = outputs[0]

        boxes = []
        confidences = []
        class_ids = []

        for det in outputs:

            obj_conf = det[4]
            if obj_conf < 1e-6:
                continue

            class_scores = det[5:]
            class_id = int(np.argmax(class_scores))
            confidence = float(obj_conf * class_scores[class_id])

            if confidence < self.conf_threshold:
                continue

            cx, cy, bw, bh = det[:4]

            x = cx - bw / 2
            y = cy - bh / 2

            x = (x - left) / scale
            y = (y - top) / scale
            bw /= scale
            bh /= scale

            x = max(0, min(orig_w - 1, x))
            y = max(0, min(orig_h - 1, y))
            bw = max(0, min(orig_w - x, bw))
            bh = max(0, min(orig_h - y, bh))

            boxes.append([int(x), int(y), int(bw), int(bh)])
            confidences.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return [], [], []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.conf_threshold,
            self.nms_threshold,
        )

        final_boxes = []
        final_confs = []
        final_ids = []

        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confs.append(confidences[i])
                final_ids.append(class_ids[i])

        return final_boxes, final_confs, final_ids

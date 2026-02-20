import os
import cv2

from src.inference.opencv_dnn_detector import OpenCVDNNDetector
from src.utils.config import load_config
from src.utils.labels import load_coco_labels
from src.utils.visualization import draw_detections


def main():

    # -----------------------------
    # Load configuration
    # -----------------------------
    config = load_config()
    model_cfg = config["model"]

    allowed_classes = set(model_cfg["allowed_classes"])
    class_thresholds = model_cfg.get("class_confidence_thresholds", {})

    # -----------------------------
    # Load model
    # -----------------------------
    detector = OpenCVDNNDetector(
        model_path=model_cfg["path"],
        input_size=model_cfg["input_size"],
        conf_threshold=model_cfg["confidence_threshold"],
        nms_threshold=model_cfg["nms_threshold"],
    )

    class_names = load_coco_labels()
    allowed_ids = {class_names.index(c) for c in allowed_classes if c in class_names}

    # -----------------------------
    # Image folder input
    # -----------------------------
    image_folder = input("Enter image folder path: ").strip()

    if not os.path.exists(image_folder):
        print("Folder not found.")
        return

    # -----------------------------
    # Process images
    # -----------------------------
    for filename in os.listdir(image_folder):

        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(image_folder, filename)
        print(f"\nTesting {filename}")

        frame = cv2.imread(path)

        boxes, confs, ids = detector.infer(frame)

        filtered_boxes = []
        filtered_confs = []
        filtered_ids = []

        for box, conf, cls_id in zip(boxes, confs, ids):

            if cls_id not in allowed_ids:
                continue

            class_name = class_names[cls_id]

            # Apply class-specific threshold if available
            threshold = class_thresholds.get(
                class_name,
                model_cfg["confidence_threshold"]
            )

            if conf >= threshold:
                filtered_boxes.append(box)
                filtered_confs.append(conf)
                filtered_ids.append(cls_id)

                print(f"Detected: {class_name} | Confidence: {conf:.3f}")

        # Draw detections
        if filtered_boxes:
            frame = draw_detections(
                frame,
                filtered_boxes,
                filtered_confs,
                filtered_ids,
                class_names,
            )

        cv2.imshow("Result", frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
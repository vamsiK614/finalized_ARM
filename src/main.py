import os
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2

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
    camera_cfg = config["camera"]
    runtime_cfg = config["runtime"]

    allowed_classes = set(model_cfg["allowed_classes"])
    class_thresholds = model_cfg.get("class_confidence_thresholds", {})

    show_fps = runtime_cfg["show_fps"]
    save_images = runtime_cfg["save_detection_images"]

    # -----------------------------
    # Initialize camera
    # -----------------------------
    if not Picamera2.global_camera_info():
        print("No camera detected.")
        return

    picam2 = Picamera2()

    cam_config = picam2.create_preview_configuration(
        main={
            "format": "RGB888",
            "size": (camera_cfg["width"], camera_cfg["height"]),
        }
    )

    picam2.configure(cam_config)
    picam2.start()
    time.sleep(2)

    # -----------------------------
    # Load model
    # -----------------------------
    detector = OpenCVDNNDetector(
        model_path=model_cfg["path"],
        input_size=model_cfg["input_size"],
        conf_threshold=model_cfg["confidence_threshold"],  # base threshold (low)
        nms_threshold=model_cfg["nms_threshold"],
    )

    class_names = load_coco_labels()
    allowed_ids = {class_names.index(c) for c in allowed_classes if c in class_names}

    # -----------------------------
    # Prepare logging
    # -----------------------------
    os.makedirs("runs", exist_ok=True)
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("runs", f"detections_{session_time}.txt")

    log_file = open(log_path, "a")
    log_file.write(f"# Session started: {datetime.now()}\n")
    log_file.flush()

    image_dir = None
    if save_images:
        image_dir = os.path.join("runs", f"session_{session_time}")
        os.makedirs(image_dir, exist_ok=True)

    print("Detection engine started")

    # -----------------------------
    # FPS tracking
    # -----------------------------
    fps_counter = 0
    fps_timer = time.time()

    frame_number = 0

    try:
        while True:

            frame = picam2.capture_array()
            frame_number += 1

            boxes, confs, ids = detector.infer(frame)

            filtered_boxes = []
            filtered_confs = []
            filtered_ids = []

            for box, conf, cls_id in zip(boxes, confs, ids):

                if cls_id not in allowed_ids:
                    continue

                class_name = class_names[cls_id]

                # Get class-specific threshold
                threshold = class_thresholds.get(
                    class_name,
                    model_cfg["confidence_threshold"]  # fallback
                )

                if conf >= threshold:
                    filtered_boxes.append(box)
                    filtered_confs.append(conf)
                    filtered_ids.append(cls_id)

            # -----------------------------
            # Log if detections exist
            # -----------------------------
            if filtered_boxes:

                entries = [
                    f"{class_names[cls_id]}({conf:.2f})"
                    for conf, cls_id in zip(filtered_confs, filtered_ids)
                ]

                combined_entry = ", ".join(entries)

                timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                log_file.write(f"{timestamp}, {combined_entry}, frame={frame_number}\n")
                log_file.flush()

                if save_images:
                    annotated = draw_detections(
                        frame.copy(),
                        filtered_boxes,
                        filtered_confs,
                        filtered_ids,
                        class_names,
                    )

                    filename = datetime.now().strftime("%H%M%S_%f") + ".jpg"
                    cv2.imwrite(os.path.join(image_dir, filename), annotated)

            # -----------------------------
            # Optional FPS display
            # -----------------------------
            if show_fps:
                fps_counter += 1
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    print(f"Inference FPS: {fps_counter / elapsed:.2f}")
                    fps_counter = 0
                    fps_timer = time.time()

    except KeyboardInterrupt:
        print("Stopping detection...")

    finally:
        picam2.stop()
        log_file.close()
        print("Clean exit")


if __name__ == "__main__":
    main()
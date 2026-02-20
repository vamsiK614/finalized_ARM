import os
import time
from datetime import datetime

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
    runtime_cfg = config["runtime"]

    allowed_classes = set(model_cfg["allowed_classes"])
    class_thresholds = model_cfg.get("class_confidence_thresholds", {})

    show_fps = runtime_cfg["show_fps"]
    save_images = runtime_cfg["save_detection_images"]

    # -----------------------------
    # Video input
    # -----------------------------
    video_path = input("Enter video path: ").strip()

    if not os.path.exists(video_path):
        print("Video not found.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("Could not determine FPS.")
        return

    duration = total_frames / fps

    print(f"Duration: {duration:.2f} seconds")
    print(f"Video FPS: {fps}")

    # -----------------------------
    # Load model
    # -----------------------------
    detector = OpenCVDNNDetector(
        model_path=model_cfg["path"],
        input_size=model_cfg["input_size"],
        conf_threshold=model_cfg["confidence_threshold"],  # base threshold
        nms_threshold=model_cfg["nms_threshold"],
    )

    class_names = load_coco_labels()
    allowed_ids = {class_names.index(c) for c in allowed_classes if c in class_names}

    # -----------------------------
    # Prepare logging
    # -----------------------------
    os.makedirs("runs", exist_ok=True)
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join("runs", f"video_detections_{session_time}.txt")
    log_file = open(log_path, "a")
    log_file.write(f"# Video session started: {datetime.now()}\n")
    log_file.flush()

    image_dir = None
    if save_images:
        image_dir = os.path.join("runs", f"video_session_{session_time}")
        os.makedirs(image_dir, exist_ok=True)

    print("Running real-time frame-drop simulation...")

    # -----------------------------
    # Real-time simulation
    # -----------------------------
    simulation_start = time.time()
    frame_index = 0

    fps_counter = 0
    fps_timer = time.time()

    while True:

        elapsed_real = time.time() - simulation_start
        expected_frame = int(elapsed_real * fps)

        # If ahead → wait
        if frame_index > expected_frame:
            time.sleep(0.001)
            continue

        # If behind → skip frames
        while frame_index < expected_frame:
            ret = cap.grab()
            if not ret:
                break
            frame_index += 1

        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # -----------------------------
        # Inference
        # -----------------------------
        boxes, confs, ids = detector.infer(frame)

        filtered_boxes = []
        filtered_confs = []
        filtered_ids = []

        for box, conf, cls_id in zip(boxes, confs, ids):

            if cls_id not in allowed_ids:
                continue

            class_name = class_names[cls_id]

            threshold = class_thresholds.get(
                class_name,
                model_cfg["confidence_threshold"]
            )

            if conf >= threshold:
                filtered_boxes.append(box)
                filtered_confs.append(conf)
                filtered_ids.append(cls_id)

        # -----------------------------
        # Logging
        # -----------------------------
        if filtered_boxes:

            entries = [
                f"{class_names[cls_id]}({conf:.2f})"
                for conf, cls_id in zip(filtered_confs, filtered_ids)
            ]

            combined_entry = ", ".join(entries)

            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            log_file.write(f"{timestamp}, {combined_entry}, frame={frame_index}\n")
            log_file.flush()

            if save_images:
                annotated = draw_detections(
                    frame.copy(),
                    filtered_boxes,
                    filtered_confs,
                    filtered_ids,
                    class_names,
                )

                filename = f"frame_{frame_index}.jpg"
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

        if elapsed_real >= duration:
            break

    cap.release()
    log_file.close()
    print("Video simulation finished cleanly.")


if __name__ == "__main__":
    main()
Edge AI Road Damage Detection (YOLOv5s + Raspberry Pi)

Real-time road damage detection system using YOLOv5s deployed on Raspberry Pi with OpenCV DNN (ONNX).

Detects:

Longitudinal crack

Transverse crack

Alligator crack

Pothole

Optimized for embedded real-time inference (~5 FPS at 256 input size).

1. Features

YOLOv5s ONNX inference using OpenCV DNN

Configurable input size

Class-specific confidence thresholds

Real-time Raspberry Pi camera support

Video file testing mode

Image folder testing mode

Optional FPS display

Optional annotated image saving

Lightweight embedded deployment

2. Project Structure
configs/
models/
src/
    main.py
    inference/opencv_dnn_detector.py
    utils/
        config.py
        labels.py
        preprocessing.py
        visualization.py
test_images.py
run_on_video.py
requirements.txt
3. Installation
3.1 Clone Repository
git clone <your_repo_url>
cd <repo_name>
3.2 Install Python Dependencies
pip install -r requirements.txt

Contents of requirements.txt:

numpy>=1.24
opencv-python>=4.8
PyYAML>=6.0
picamera2>=0.3.12
3.3 Alternative Installation for Raspberry Pi (Recommended)

On Raspberry Pi OS, it is recommended to install picamera2 using apt:

sudo apt update
sudo apt install python3-picamera2

Then install remaining Python packages:

pip install numpy opencv-python PyYAML

This ensures full compatibility with libcamera.

4. Configuration

All runtime behavior is controlled via:

configs/model_config.yaml

Example:

model:
  path: models/yolov5/yolov5s.onnx
  input_size: 256
  confidence_threshold: 0.5
  nms_threshold: 0.45

  allowed_classes:
    - longitudinal crack
    - transverse crack
    - aligator crack
    - pothole

  class_confidence_thresholds:
    longitudinal crack: 0.6
    transverse crack: 0.6
    aligator crack: 0.65
    pothole: 0.7

camera:
  width: 480
  height: 360

runtime:
  show_fps: true
  save_detection_images: true
5. Runtime Options

You can disable optional features directly in model_config.yaml.

Disable FPS Display
runtime:
  show_fps: false
Disable Annotated Image Saving
runtime:
  save_detection_images: false

When disabled:

No FPS printing

No image writing

Maximum inference performance

Logging still occurs when detections are present.

6. Running the System
6.1 Raspberry Pi Camera (Production Mode)
python -m src.main

Headless mode

Logs detections

Saves annotated images (if enabled)

6.2 Test on Video File
python run_on_video.py

Enter video path when prompted.

6.3 Test on Image Folder
python test_images.py

Enter folder path when prompted.

7. Performance

Deployment configuration:

Model: YOLOv5s

Input size: 256

Backend: OpenCV DNN (CPU)

Platform: Raspberry Pi

Average performance: ~5 FPS

Performance depends on:

Input resolution

Camera resolution

Enabled runtime options

Raspberry Pi model

8. Important Notes
8.1 Raspberry Pi Only for Camera Mode

src/main.py requires:

Raspberry Pi hardware

libcamera

picamera2

test_images.py and run_on_video.py work on any machine.

8.2 Class Names Must Match Model

Entries in allowed_classes and class_confidence_thresholds must exactly match the class names used during training.

Spelling must be identical.

8.3 Input Size Must Match Exported Model

If you export your model with:

--img 256

You must set:

input_size: 256

in model_config.yaml.

9. Optimization Notes

Performance improvements were achieved by:

Using YOLOv5s instead of larger variants

Reducing input size from 640 to 256

Using OpenCV DNN CPU backend

Disabling unnecessary display windows

Class-specific threshold tuning

Configuration-driven runtime control

10. Recommended Hardware

Raspberry Pi 4 (4GB+ recommended)

Raspberry Pi Camera Module

Raspberry Pi OS (64-bit recommended)
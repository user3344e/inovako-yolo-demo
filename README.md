# Inovako YOLO Demo

This is a small YOLO demo I prepared for the Inovako AI intern technical task.

The project uses a pretrained YOLOv8n model with OpenCV to run real-time object detection from a webcam. I did not train a custom model in this demo; the main purpose was to understand the basic object detection flow and test YOLO on a live camera input.

## What it does

- opens the webcam
- runs YOLO detection on each frame
- draws bounding boxes and labels
- shows the FPS value
- saves the current frame when `s` is pressed
- exits when `q` is pressed

## Files

```text
main.py              main Python script
requirements.txt     required packages
outputs/             saved output image
report.pdf           research report
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

If needed on Windows:

```bash
py main.py
```

## Output

The saved example result is in:

```text
outputs/yolo_result.png
```

## Notes

The demo uses `yolov8n.pt`, which is a lightweight pretrained YOLO model. It is fast enough for a simple webcam test, but it can still make mistakes in some scenes. For a real application, the model output should be filtered and tested more carefully.

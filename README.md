\# Inovako YOLO Demo



This project demonstrates a simple real-time object detection application using a pretrained YOLO model and OpenCV.



\## Purpose



The aim of this project is to understand the basic workflow of a real-time object detection system. The application captures frames from a webcam, sends each frame to a pretrained YOLO model, filters detections using simple rules, and visualizes the detected objects with bounding boxes, class labels, confidence scores, and FPS information.



\## Technologies Used



\- Python

\- Ultralytics YOLO

\- OpenCV

\- Pretrained YOLOv8n model



\## Project Structure



```text

inovako-yolo-demo/

│

├── main.py

├── requirements.txt

├── README.md

├── .gitignore

├── outputs/

│   └── yolo\_result.png

└── report.pdf

```



\## Installation



First, install the required Python packages:



```bash

pip install -r requirements.txt

```



\## Run



To run the application:



```bash

python main.py

```



If Python is configured as `py` on Windows, this can also be used:



```bash

py main.py

```



\## How It Works



1\. The webcam is opened using OpenCV.

2\. Frames are captured continuously from the camera.

3\. Each frame is passed to the pretrained YOLO model.

4\. YOLO performs object detection on the frame.

5\. The model returns bounding boxes, class labels, and confidence scores.

6\. Basic filtering is applied using confidence threshold and minimum bounding box area.

7\. The detection results are drawn on the frame.

8\. The FPS value is displayed on the screen.

9\. Press `s` to save the current output image.

10\. Press `q` to exit the application.



\## Output



An example output is provided in the `outputs/` folder.



The output shows that the model can detect objects in real time using the webcam. The FPS value is displayed to observe real-time performance.



\## Notes



This project does not train a new model from scratch. It uses a pretrained YOLOv8n model for inference.



The main goal of this demo is to understand the basic object detection pipeline:



```text

Webcam frame

&#x20;     ↓

YOLO model

&#x20;     ↓

Bounding box + class label + confidence score

&#x20;     ↓

Filtering

&#x20;     ↓

Visualization with OpenCV

```



This simple pipeline represents the foundation of many real-time computer vision systems.




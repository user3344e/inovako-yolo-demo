import os
os.environ["YOLO_VERBOSE"] = "False"

import io
import time
import cv2
import logging
from contextlib import redirect_stdout, redirect_stderr

from ultralytics import YOLO
from ultralytics.utils import LOGGER


LOGGER.disabled = True
LOGGER.setLevel(logging.CRITICAL)

MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.65
MIN_BOX_AREA_RATIO = 0.02
PRINT_INTERVAL = 1.0
OUTPUT_PATH = "outputs/yolo_result.png"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

previous_time = time.time()
last_print_time = time.time()

print("YOLO real-time object detection demo started.")
print("Press 's' to save the current frame.")
print("Press 'q' to quit.")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame could not be read.")
        break

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            verbose=False
        )

    result = results[0]
    filtered_detections = []
    annotated_frame = frame.copy()

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        box_area_ratio = box_area / frame_area

        if box_area_ratio < MIN_BOX_AREA_RATIO:
            continue

        detection = {
            "class": class_name,
            "confidence": confidence,
            "box": [x1, y1, x2, y2],
            "area_ratio": box_area_ratio
        }

        filtered_detections.append(detection)

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    current_time = time.time()
    elapsed_time = max(current_time - previous_time, 1e-6)
    fps = 1 / elapsed_time
    previous_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"conf >= {CONF_THRESHOLD}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    if current_time - last_print_time >= PRINT_INTERVAL:
        print("\n--- Detection Summary ---")

        if not filtered_detections:
            print("No reliable detection.")

        for detection in filtered_detections:
            print(
                f"{detection['class']} | "
                f"confidence: {detection['confidence']:.2f} | "
                f"area ratio: {detection['area_ratio']:.3f}"
            )

        last_print_time = current_time

    cv2.imshow("YOLO Object Detection Demo", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite(OUTPUT_PATH, annotated_frame)
        print(f"Image saved: {OUTPUT_PATH}")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import os
os.environ["YOLO_VERBOSE"] = "False"

import logging
import io
import time
import cv2
from contextlib import redirect_stdout, redirect_stderr

from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Ultralytics loglarini sustur
LOGGER.disabled = True
LOGGER.setLevel(logging.CRITICAL)

# Pretrained YOLO modeli
# yolov8n.pt hizli ve hafiftir. Daha yuksek dogruluk icin yolov8s.pt denenebilir.
model = YOLO("yolov8n.pt")

# Kamera ac
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera acilamadi.")
    exit()

# Ayarlar
CONF_THRESHOLD = 0.65
MIN_BOX_AREA_RATIO = 0.02
PRINT_INTERVAL = 1.0

prev_time = time.time()
last_print_time = time.time()

print("YOLO real-time object detection demo basladi.")
print("Cikmak icin q, goruntu kaydetmek icin s tusuna basin.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame okunamadi.")
        break

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    # YOLO inference
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            verbose=False
        )

    result = results[0]
    filtered_detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        box_area_ratio = box_area / frame_area

        # Cok kucuk kutulari ele
        if box_area_ratio < MIN_BOX_AREA_RATIO:
            continue

        filtered_detections.append({
            "class": class_name,
            "confidence": confidence,
            "box": [x1, y1, x2, y2],
            "area_ratio": box_area_ratio
        })

    annotated_frame = result.plot()

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

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

        if len(filtered_detections) == 0:
            print("No reliable detection.")

        for det in filtered_detections:
            print(
                f"{det['class']} | "
                f"confidence: {det['confidence']:.2f} | "
                f"area ratio: {det['area_ratio']:.3f}"
            )

        last_print_time = current_time

    cv2.imshow("YOLO Object Detection - Filtered Demo", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    # s tusuna basinca goruntuyu kaydet
    if key == ord("s"):
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite("outputs/yolo_result.png", annotated_frame)
        print("Goruntu kaydedildi: outputs/yolo_result.png")

    # q tusuna basinca cik
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import cvzone
import math
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# COCO classes
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "mouse", "earphones"
]

# Load Haar Cascade for number plate detection
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
if plate_cascade.empty():
    print("❌ Haar cascade file not found! Put haarcascade_russian_plate_number.xml in this folder.")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

while True:
    success, img = cap.read()
    if not success:
        break

    # Run YOLO
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Draw YOLO detection
            cvzone.cornerRect(img, (x1, y1, w, h), l=30, t=5, colorR=(255, 0, 0))
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                               scale=1, thickness=2, colorR=(255, 0, 0))

            # If it's a car → run number plate detection
            if classNames[cls] == "car":
                car_roi = img[y1:y2, x1:x2]
                if car_roi.size > 0:  # Ensure ROI is valid
                    gray_car = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
                    plates = plate_cascade.detectMultiScale(gray_car, scaleFactor=1.1,
                                                            minNeighbors=4, minSize=(30, 10))
                    for (px, py, pw, ph) in plates:
                        cv2.rectangle(img, (x1 + px, y1 + py), (x1 + px + pw, y1 + py + ph),
                                      (0, 255, 0), 2)
                        cv2.putText(img, 'Number Plate', (x1 + px, y1 + py - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO + Plate Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

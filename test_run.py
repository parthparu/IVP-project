import cv2
import cvzone
import math
import os
from ultralytics import YOLO
import easyocr

# Load YOLO models
car_model = YOLO('yolov8n.pt')                 # COCO model for car detection
plate_model = YOLO('yolov8n-license-plate.pt') # custom/trained model for plates

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
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# EasyOCR reader
reader = easyocr.Reader(['en'])

# Path to folder containing images
folder_path = "C:\\Users\\sanga\\OneDrive\\Desktop\\opencv"   # change this
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Preprocessing for OCR
def preprocess_plate(plate_roi):
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    resized = cv2.resize(morph, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized

for file in image_files:
    img_path = os.path.join(folder_path, file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Could not read image {file}")
        continue

    # Run YOLO for car detection
    results = car_model(img, stream=True, conf=0.5)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if classNames[cls] == "car" and conf > 0.5:
                cvzone.cornerRect(img, (x1, y1, w, h), l=30, t=5, colorR=(255, 0, 0))
                cvzone.putTextRect(img, f'car {conf:.2f}', (x1, y1-10), scale=1, thickness=2, colorR=(255, 0, 0))

                # Crop car ROI
                car_roi = img[y1:y2, x1:x2]
                if car_roi.size == 0:
                    continue

                # Detect plate inside car ROI
                plate_results = plate_model(car_roi, stream=True, conf=0.5)

                for pr in plate_results:
                    for pbox in pr.boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                        pw, ph = px2 - px1, py2 - py1

                        cv2.rectangle(img, (x1+px1, y1+py1), (x1+px2, y1+py2), (0,255,0), 2)

                        # Extract plate ROI
                        plate_roi = car_roi[py1:py2, px1:px2]
                        if plate_roi.size == 0:
                            continue

                        processed_plate = preprocess_plate(plate_roi)

                        # OCR with allowlist
                        results_ocr = reader.readtext(processed_plate, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        if results_ocr:
                            text = results_ocr[0][1]
                            print(f"📷 {file} → Detected Plate: {text}")
                            cv2.putText(img, text, (x1+px1, y1+py1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("YOLO + Plate Detection + EasyOCR", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

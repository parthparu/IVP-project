import cv2
import cvzone
import math
import pytesseract
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox

# If using Windows, set Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# COCO class names
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

# Load Haar Cascade for number plate detection
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
if plate_cascade.empty():
    messagebox.showerror("Error", "Haar cascade file not found!")
    exit()

def detect_and_display(img):
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
                if car_roi.size > 0:
                    gray_car = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
                    plates = plate_cascade.detectMultiScale(gray_car, scaleFactor=1.1,
                                                            minNeighbors=4, minSize=(30, 10))
                    for (px, py, pw, ph) in plates:
                        # Draw rectangle for plate
                        cv2.rectangle(img, (x1 + px, y1 + py), (x1 + px + pw, y1 + py + ph),
                                      (0, 255, 0), 2)

                        # Extract plate ROI
                        plate_roi = car_roi[py:py+ph, px:px+pw]

                        # Preprocess for OCR
                        plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                        _, plate_thresh = cv2.threshold(plate_gray, 120, 255, cv2.THRESH_BINARY)

                        # OCR
                        text = pytesseract.image_to_string(plate_thresh, config='--psm 8')
                        text = text.strip()

                        # Show OCR text
                        cv2.putText(img, text, (x1 + px, y1 + py - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        print("Detected Plate:", text)

    return img

def run_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam!")
        return

    while True:
        success, img = cap.read()
        if not success:
            break
        img = detect_and_display(img)
        cv2.imshow("YOLO + Plate OCR (Webcam)", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not filepath:
        return
    img = cv2.imread(filepath)
    if img is None:
        messagebox.showerror("Error", "Could not read image!")
        return
    img = detect_and_display(img)
    cv2.imshow("YOLO + Plate OCR (Image)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Tkinter UI
root = tk.Tk()
root.title("Detection UI")
root.geometry("300x200")

tk.Label(root, text="Choose Detection Mode:", font=("Arial", 14)).pack(pady=20)
tk.Button(root, text="📷 Webcam Detection", font=("Arial", 12), command=run_webcam).pack(pady=10)
tk.Button(root, text="🖼️ Image Detection", font=("Arial", 12), command=run_image).pack(pady=10)

root.mainloop()

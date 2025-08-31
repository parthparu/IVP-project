import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

haarcascade = "haarcascade_russian_plate_number.xml"

img = cv2.imread("car.jpg")

if img is None:
    print("Image not found!")
    exit()

min_area = 500

plate_cascade = cv2.CascadeClassifier(haarcascade)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

for (x, y, w, h) in plates:
    area = w * h
    if area > min_area:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Number Plate", (x, y - 5), 1, 1, (255, 0, 255), 2)

        plate_roi = img[y:y+h, x:x+w]
        cv2.imshow("Plate ROI", plate_roi)

        # Preprocess for better OCR accuracy
        gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        filtered_plate = cv2.bilateralFilter(gray_plate, 11, 17, 17)
        thresh_plate = cv2.adaptiveThreshold(
            filtered_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        resized_plate = cv2.resize(thresh_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # OCR extraction
        plate_text = pytesseract.image_to_string(resized_plate, config='--psm 8')
        print("Detected Plate Text:", plate_text)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

haarcascade = "haarcascade_russian_plate_number.xml"

img = cv2.imread("car.jpg")

if img is None:
    print("Image not found!")
    exit()

min_area = 500

def preprocess_plate(plate_roi):
    """
    Enhanced preprocessing pipeline for better OCR accuracy
    """
    # Convert to grayscale if not already
    if len(plate_roi.shape) == 3:
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_roi.copy()
    
    # Resize image for better OCR (upscale small plates)
    height, width = gray.shape
    if height < 50 or width < 200:
        scale_factor = max(50/height, 200/width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply sharpening kernel
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    sharpened = cv2.filter2D(filtered, -1, kernel)
    
    # Morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
    
    # Apply adaptive thresholding for better text extraction
    # Try multiple threshold methods and pick the best result
    thresh_methods = [
        cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
        cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ]
    
    return thresh_methods, cleaned

def extract_text_with_multiple_configs(processed_images):
    """
    Try multiple OCR configurations and return the best result
    """
    # Different OCR configurations to try
    configs = [
        '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        '--psm 8',
        '--psm 7',
        '--psm 13'
    ]
    
    best_text = ""
    best_confidence = 0
    
    for img in processed_images:
        for config in configs:
            try:
                # Get text with confidence scores
                data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                text = pytesseract.image_to_string(img, config=config).strip()
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Filter out results that are too short or have low confidence
                if len(text) >= 5 and avg_confidence > best_confidence:
                    best_text = text
                    best_confidence = avg_confidence
                    
            except Exception as e:
                continue
    
    return best_text, best_confidence

plate_cascade = cv2.CascadeClassifier(haarcascade)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

for (x, y, w, h) in plates:
    area = w * h
    if area > min_area:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Number Plate", (x, y - 5), 1, 1, (255, 0, 255), 2)

        # Extract the plate region with some padding
        padding = 5
        y_start = max(0, y - padding)
        y_end = min(img.shape[0], y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(img.shape[1], x + w + padding)
        
        plate_roi = img[y_start:y_end, x_start:x_end]
        
        # Apply enhanced preprocessing
        processed_images, cleaned = preprocess_plate(plate_roi)
        
        # Extract text using multiple configurations
        plate_text, confidence = extract_text_with_multiple_configs(processed_images)
        
        # Clean up the extracted text
        plate_text = ''.join(char for char in plate_text if char.isalnum() or char.isspace())
        plate_text = ' '.join(plate_text.split())  # Remove extra whitespace
        
        print(f"Detected Plate Text: '{plate_text}'")
        print(f"Confidence: {confidence:.2f}%")
        
        # Display original and processed images
        cv2.imshow("Original Plate ROI", plate_roi)
        cv2.imshow("Cleaned Plate ROI", cleaned)
        cv2.imshow("Best Threshold", processed_images[0])

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
ext = pytesseract.image_to_string(plate_thresh, config="--psm 8")
        # print("Detected License Plate Number:", text.strip())
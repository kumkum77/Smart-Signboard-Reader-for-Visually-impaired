# test_ocr.py
import cv2
from PIL import Image
import pytesseract
import os

# 1️⃣ Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 2️⃣ Load the image (use raw string to avoid path errors)
image_path = r"C:\Users\kumku\Downloads\VIT Downloads\cv_project\preprocessed_dataset\image_759.jpg"

if not os.path.exists(image_path):
    print("Image file not found. Check the path!")
    exit()

# 3️⃣ Read image using OpenCV
img = cv2.imread(image_path)

# 4️⃣ Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 5️⃣ Apply thresholding to improve contrast
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Optional: Denoise / remove noise
thresh = cv2.medianBlur(thresh, 3)

# 6️⃣ Save temporary image for pytesseract
temp_path = "temp_processed.jpg"
cv2.imwrite(temp_path, thresh)

# 7️⃣ OCR extraction
# --psm 6 assumes a single uniform block of text
text = pytesseract.image_to_string(Image.open(temp_path), config="--psm 6", lang="eng")

print("Extracted Text:")
print(text)

# 8️⃣ Clean up temporary file
os.remove(temp_path)
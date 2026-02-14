import os
import cv2
import numpy as np
import hashlib
import shutil

# 1. Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REJECT_DIR = os.path.join(BASE_DIR, "data", "rejected")

# Load Face Detector (Pre-trained model included in OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_image_hash(image):
    """Generates a fingerprint for the image to find duplicates."""
    return hashlib.md5(image.tobytes()).hexdigest()

def is_blurry_or_flat(image, threshold=50):
    """
    Checks if image has low texture (often cartoons/drawings are 'flat').
    Real trash is messy and has high variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def has_face(image):
    """Checks if there is a human face in the photo."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

def clean_dataset():
    if not os.path.exists(REJECT_DIR):
        os.makedirs(REJECT_DIR)

    print(f"🧹 Starting Deep Clean...")
    print(f"🗑️  'Bad' images will be moved to: {REJECT_DIR}")

    total_moved = 0
    seen_hashes = set()

    # Loop through Plastic, Glass, Metal, Cardboard, Biowaste
    classes = ["plastic", "glass", "metal", "cardboard", "biowaste"]
    
    for class_name in classes:
        folder_path = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(folder_path): continue

        print(f"\n📂 Cleaning {class_name.upper()}...")
        files = os.listdir(folder_path)
        
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            
            # 1. Try to open image
            try:
                img = cv2.imread(file_path)
                if img is None:
                    # Corrupt file
                    shutil.move(file_path, os.path.join(REJECT_DIR, f"corrupt_{filename}"))
                    continue
            except:
                continue

            # 2. Check Duplicates
            img_hash = get_image_hash(img)
            if img_hash in seen_hashes:
                print(f"   found duplicate: {filename}")
                shutil.move(file_path, os.path.join(REJECT_DIR, f"dup_{filename}"))
                total_moved += 1
                continue
            seen_hashes.add(img_hash)

            # 3. Check for Faces (Stock photos of people)
            if has_face(img):
                print(f"   found face: {filename}")
                shutil.move(file_path, os.path.join(REJECT_DIR, f"face_{filename}"))
                total_moved += 1
                continue

            # 4. Check for 'Flat' images (Cartoons/Drawings often have low variance)
            # We skip this for Biowaste because dirt/compost can be blurry.
            if class_name != "biowaste" and is_blurry_or_flat(img, threshold=20):
                print(f"   found blurry/drawing: {filename}")
                shutil.move(file_path, os.path.join(REJECT_DIR, f"blur_{filename}"))
                total_moved += 1
                continue

    print(f"\n✨ Done! Moved {total_moved} images to the 'rejected' folder.")
    print("👉 Please quickly check the 'rejected' folder to make sure no good trash was lost.")

if __name__ == "__main__":
    clean_dataset()
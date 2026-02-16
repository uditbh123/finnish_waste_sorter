import os
import cv2
import glob
import random
import numpy as np
import albumentations as A
from pathlib import Path

# 1. Setup Paths 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# 2. The "Color Jitter" Pipeline
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    
    # Color Jitter: Fixes "Orange = Bio"
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),

    # Texture Noise: Fixes "Smooth = Plastic"
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), 

    A.Perspective(scale=(0.05, 0.1), p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
    
    # Updated CoarseDropout syntax
    A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(10, 20), hole_width_range=(10, 20), p=0.3)
])

def augment_specific_files(folder_name, keywords, target_count=50):
    folder_path = os.path.join(PROCESSED_DIR, folder_name)
    if not os.path.exists(folder_path): return

    all_files = glob.glob(os.path.join(folder_path, "*"))
    
    # Filter: Must match keywords AND be an original (not already augmented)
    target_files = [
        f for f in all_files 
        if any(k in os.path.basename(f).lower() for k in keywords) 
        and "_aug_" not in os.path.basename(f)
        and "_mosaic_" not in os.path.basename(f)
    ]

    print(f"\n🔍 Found {len(target_files)} ORIGINAL images in '{folder_name}' matching {keywords}.")

    for file_path in target_files:
        image = cv2.imread(file_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filename = Path(file_path).stem

        for i in range(target_count):
            augmented = transform(image=image)['image']
            save_name = f"{filename}_aug_{i:03d}.jpg"
            save_path = os.path.join(folder_path, save_name)
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

def create_mosaic(class_name, num_mosaics=100):
    folder_path = os.path.join(PROCESSED_DIR, class_name)
    if not os.path.exists(folder_path): return
    
    all_files = glob.glob(os.path.join(folder_path, "*"))
    source_images = [f for f in all_files if "_aug_" not in f and "_mosaic_" not in f]
    
    if len(source_images) < 4:
        print(f"⚠️  Skipping Mosaic for {class_name}: Need at least 4 original images.")
        return

    print(f"🧩 Creating {num_mosaics} mosaic piles for: {class_name.upper()}...")

    for i in range(num_mosaics):
        choices = random.sample(source_images, 4)
        imgs = []
        for path in choices:
            img = cv2.imread(path)
            if img is None: continue
            img = cv2.resize(img, (112, 112))
            imgs.append(img)
            
        if len(imgs) < 4: continue

        top_row = np.hstack((imgs[0], imgs[1]))
        bot_row = np.hstack((imgs[2], imgs[3]))
        mosaic = np.vstack((top_row, bot_row))
        
        save_name = f"mosaic_pile_{i:03d}.jpg"
        cv2.imwrite(os.path.join(folder_path, save_name), mosaic)

if __name__ == "__main__":
    print("🚀 Starting Hybrid Augmentation...")

    # 1. YOUR ORIGINAL PHOTOS (Target Brands)
    augment_specific_files("cardboard", ["valio", "milk", "maito", "515W"], target_count=50)
    augment_specific_files("plastic", ["atria", "meat", "jauheliha", "coca", "cola"], target_count=50)
    
    # 2. THE NEW WEB DATA (Multiplying the 40 downloads -> 800 images)
    augment_specific_files("plastic", ["web_"], target_count=20)
    augment_specific_files("glass", ["web_"], target_count=20)
    augment_specific_files("metal", ["web_"], target_count=20)
    augment_specific_files("cardboard", ["web_"], target_count=20)

    # 3. BIOWASTE (Just a little variation)
    augment_specific_files("biowaste", ["jpg", "jpeg", "png"], target_count=1)

    # 4. MOSAICS (Fixes "Pile" Error)
    create_mosaic("plastic", num_mosaics=200)
    create_mosaic("cardboard", num_mosaics=100)
    create_mosaic("metal", num_mosaics=100) 
    create_mosaic("glass", num_mosaics=100)

    # 5. HARD NEGATIVE MINING (The "Fix" for Tricky Images)
    # These match the files you just moved from test_dump
    print("\n🔨 Starting Hard Negative Mining...")
    augment_specific_files("plastic", ["jsdsd", "ssas", "test2"], target_count=50)
    augment_specific_files("glass", ["asslg"], target_count=50)
    augment_specific_files("metal", ["download12"], target_count=50)
    # We add variations for the tricky bio image too
    augment_specific_files("biowaste", ["download"], target_count=20)

    print("\n✅ Done! Dataset is massive and ready.")
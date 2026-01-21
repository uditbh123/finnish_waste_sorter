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
# 🟢 THIS IS THE FIX: We shift Hue, Saturation, and Brightness.
# This ensures "Orange" doesn't always mean Biowaste.
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    
    # Change HUE (Color), SATURATION (Intensity), and VALUE (Brightness)
    # This is the direct Albumentations equivalent of "Color Jitter"
    A.HueSaturationValue(
        hue_shift_limit=20, # Shifts colors (e.g., Orange -> Red/Yellow)
        sat_shift_limit=30, 
        val_shift_limit=20, 
        p=1.0  # Apply to EVERY augmented image
    ),

    # Randomly shift RGB channels independently for extra variation
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),

    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),

    # Texture Noise: Fixes "Smooth = Plastic" (Helps Valio Carton)
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), 

    A.Perspective(scale=(0.05, 0.1), p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3)
])

def augment_specific_files(folder_name, keywords, target_count=50):
    folder_path = os.path.join(PROCESSED_DIR, folder_name)
    if not os.path.exists(folder_path):
        return

    all_files = glob.glob(os.path.join(folder_path, "*"))

    # 🛡️ SAFETY CHECK: 
    # I filter the list to find ONLY original files.
    # 1. Must match keywords (valio, atria, etc.)
    # 2. Must NOT contain "_aug_" (so I don't copy a copy)
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

        # Convert to RGB, then Augment
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filename = Path(file_path).stem

        for i in range(target_count):
            augmented = transform(image=image)['image']
            save_name = f"{filename}_aug_{i:03d}.jpg"
            save_path = os.path.join(folder_path, save_name)
            # save as BGR (OpenCV standard)
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

def create_mosaic(class_name, num_mosaics=100):
    """
    Takes 4 random images and stitches them into a 2x2 grid.
    This teaches the model that 'Clutter' is NOT always Biowaste.
    """
    folder_path = os.path.join(PROCESSED_DIR, class_name)
    if not os.path.exists(folder_path): return
    
    # Get all clean images (no augs, no existing mosaics)
    all_files = glob.glob(os.path.join(folder_path, "*"))
    source_images = [f for f in all_files if "_aug_" not in f and "_mosaic_" not in f]
    
    if len(source_images) < 4:
        print(f"⚠️  Skipping Mosaic for {class_name}: Need at least 4 original images.")
        return

    print(f"🧩 Creating {num_mosaics} mosaic piles for: {class_name.upper()}...")

    for i in range(num_mosaics):
        # Pick 4 random images
        choices = random.sample(source_images, 4)
        
        imgs = []
        for path in choices:
            img = cv2.imread(path)
            if img is None: continue
            # Resize to 112x112 (Half of 224)
            img = cv2.resize(img, (112, 112))
            imgs.append(img)
            
        if len(imgs) < 4: continue

        # Stitch: Top Row + Bottom Row
        top_row = np.hstack((imgs[0], imgs[1]))
        bot_row = np.hstack((imgs[2], imgs[3]))
        mosaic = np.vstack((top_row, bot_row))
        
        # Save
        save_name = f"mosaic_pile_{i:03d}.jpg"
        cv2.imwrite(os.path.join(folder_path, save_name), mosaic)

if __name__ == "__main__":
    print("🚀 Starting Color Jitter Augmentation...")

    # 1. VALIO / MILK (Cardboard) -> 50 copies each
    augment_specific_files("cardboard", ["valio", "milk", "maito", "515W"], target_count=50)

    # 2. ATRIA / COKE (Plastic) -> 50 copies each
    augment_specific_files("plastic", ["atria", "meat", "jauheliha", "coca", "cola"], target_count=50)

    # 3. BIOWASTE (The Fix) -> 1 copy each
    # Since I have 500 originals, this adds 500 color-shifted versions. Total = 1000.
    augment_specific_files("biowaste", ["jpg", "jpeg", "png"], target_count=1)
    
    # --- PART 2: FIX "PILE" ERROR (The New Stuff) ---
    # We create artificial piles for non-bio classes
    create_mosaic("plastic", num_mosaics=200)   # Fixes "Many bottles = Bio"
    create_mosaic("cardboard", num_mosaics=100) # Fixes "Many boxes = Bio"
    create_mosaic("metal", num_mosaics=50)      # Fixes "Many cans = Bio"

    print("\n✅ Done!")
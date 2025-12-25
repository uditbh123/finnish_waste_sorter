import os
import cv2
import glob
import albumentations as A
from pathlib import Path

# 1. Setup Paths 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# 2. Define the "Semantic Destruction" Pipeline
# The "Generalization" Pipeline
# This forces the AI to ignore brand colors and focus on SHAPE.
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    
    # 🎨 AGGRESSIVE COLOR SHIFTING
    # This spins the color wheel wildly. Red becomes Blue, Green, Purple.
    # p=1.0 means we do this to EVERY augmented image.
    A.HueSaturationValue(hue_shift_limit=100, sat_shift_limit=50, val_shift_limit=40, p=1.0),
    
    # ⚫⚪ GRAYSCALE (20% chance)
    # Sometimes we remove color entirely so it MUST look at shape.
    A.ToGray(p=0.2),

    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
    
    # Geometric distortion (Stretches the box shape slightly)
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
    
    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3)
])

def augment_specific_files(folder_name, keywords, target_count=50):
    folder_path = os.path.join(PROCESSED_DIR, folder_name)
    if not os.path.exists(folder_path):
        print(f"⚠️  Skipping '{folder_name}' - Folder not found.")
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
    ]

    print(f"\n🔍 Found {len(target_files)} ORIGINAL images in '{folder_name}' to augment.")

    if not target_files:
        print(f"   (No new original files found in {folder_name})")
        return
    
    for file_path in target_files:
        image = cv2.imread(file_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        filename = Path(file_path).stem

        print(f"   Generating {target_count} variants for: {filename}")
        for i in range(target_count):
            augmented = transform(image=image)['image']
            
            # Save with _aug_ tag
            save_name = f"{filename}_aug_{i:03d}.jpg"
            save_path = os.path.join(folder_path, save_name)
            
            # Save
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    print("🚀 Starting Safe Data Augmentation...")

    # 1. VALIO / MILK (Cardboard) -> 50 copies each
    augment_specific_files("cardboard", ["valio", "milk", "maito", "515W"], target_count=50)

    # 2. ATRIA / COKE (Plastic) -> 50 copies each
    augment_specific_files("plastic", ["atria", "meat", "jauheliha", "coca", "cola"], target_count=50)

    # 3. BIOWASTE (The Fix) -> 3 copies each
    # This multiplies your 500 bio images by 3 to get ~1500-2000 total.
    # I use "jpg/jpeg/png" keywords to grab ALL biowaste images.
    #augment_specific_files("biowaste", ["jpg", "jpeg", "png"], target_count=2)
    
    #print("\n✅ Augmentation Complete!")
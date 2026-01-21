"""
Smart Class Balancing: Augment Small Classes to Match Largest Class
This fixes the severe imbalance (410 metal vs 1201 cardboard)
"""

import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# AGGRESSIVE augmentation for minority classes
STRONG_AUGMENTATION = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    
    # 🔥 COLOR JITTER (The key to breaking color bias)
    A.HueSaturationValue(
        hue_shift_limit=30,      # More aggressive than before
        sat_shift_limit=40, 
        val_shift_limit=30, 
        p=1.0  # Apply to EVERY augmented image
    ),
    
    A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    
    # Geometric transforms
    A.ShiftScaleRotate(
        shift_limit=0.1, 
        scale_limit=0.2, 
        rotate_limit=45,  # Full rotation
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.8
    ),
    
    A.Perspective(scale=(0.05, 0.15), p=0.5),
    
    # Simulate photo quality issues
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.GaussNoise(var_limit=(10, 50), p=1.0),
    ], p=0.3),
    
    A.CoarseDropout(
        max_holes=8, 
        max_height=30, 
        max_width=30, 
        fill_value=0,
        p=0.4
    )
])

def count_images(folder_path):
    """Count valid images in a folder"""
    return len(list(folder_path.glob("*.jpg")) + 
               list(folder_path.glob("*.png")) +
               list(folder_path.glob("*.jpeg")))

def augment_to_target(folder_path, target_count):
    """
    Augment images in folder until reaching target_count
    """
    current_count = count_images(folder_path)
    
    if current_count >= target_count:
        print(f"   ✅ {folder_path.name}: Already has {current_count} images (target: {target_count})")
        return
    
    needed = target_count - current_count
    print(f"   🔄 {folder_path.name}: Need {needed} more images (current: {current_count})")
    
    # Get all original images (exclude previously augmented)
    all_images = [f for f in folder_path.glob("*") 
                  if f.suffix.lower() in ['.jpg', '.png', '.jpeg']
                  and '_aug_' not in f.stem]  # Only use originals
    
    if len(all_images) == 0:
        print(f"   ❌ ERROR: No original images found in {folder_path.name}")
        return
    
    print(f"      Using {len(all_images)} original images as source")
    
    # Calculate how many augmentations per image
    augmentations_per_image = needed // len(all_images) + 1
    
    generated = 0
    for img_path in all_images:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generate augmented versions
        for i in range(augmentations_per_image):
            if generated >= needed:
                break
            
            augmented = STRONG_AUGMENTATION(image=img)['image']
            
            # Save with unique name
            save_name = f"{img_path.stem}_balanced_{generated:04d}.jpg"
            save_path = folder_path / save_name
            
            cv2.imwrite(str(save_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            generated += 1
        
        if generated >= needed:
            break
    
    print(f"      ✅ Generated {generated} new images")

def balance_all_classes():
    """
    Balance all classes to match the largest class
    """
    print("\n📊 STEP 1: Analyzing Current Distribution")
    print("="*60)
    
    class_counts = {}
    classes = ["biowaste", "cardboard", "glass", "metal", "plastic"]
    
    for class_name in classes:
        folder_path = PROCESSED_DIR / class_name
        if folder_path.exists():
            count = count_images(folder_path)
            class_counts[class_name] = count
            print(f"   {class_name:12s}: {count:4d} images")
        else:
            print(f"   {class_name:12s}: ❌ FOLDER NOT FOUND")
            class_counts[class_name] = 0
    
    # Find target (largest class)
    target_count = max(class_counts.values())
    print(f"\n🎯 Target count (largest class): {target_count} images")
    
    # Calculate total augmentations needed
    total_needed = sum([max(0, target_count - count) for count in class_counts.values()])
    print(f"📈 Total images to generate: {total_needed}")
    
    print("\n" + "="*60)
    print("🔄 STEP 2: Augmenting Minority Classes")
    print("="*60)
    
    for class_name, count in class_counts.items():
        if count == 0:
            continue
        
        folder_path = PROCESSED_DIR / class_name
        augment_to_target(folder_path, target_count)
    
    # Verify final counts
    print("\n" + "="*60)
    print("📊 STEP 3: Final Verification")
    print("="*60)
    
    for class_name in classes:
        folder_path = PROCESSED_DIR / class_name
        if folder_path.exists():
            final_count = count_images(folder_path)
            print(f"   {class_name:12s}: {final_count:4d} images")
    
    print("\n✅ BALANCING COMPLETE!")
    print("\n💡 Next steps:")
    print("   1. Run: python src/check_data.py (verify structure)")
    print("   2. Run: python src/train_model.py (retrain with balanced data)")

if __name__ == "__main__":
    print("🎯 SMART CLASS BALANCING")
    print("This will augment minority classes to match the largest class")
    print("="*60)
    
    response = input("Continue? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        balance_all_classes()
    else:
        print("❌ Cancelled.")
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Configuration
MODEL_PATH = "models/phase2_finetuned.keras"
CLASS_NAMES = ['biowaste', 'cardboard', 'glass', 'metal', 'plastic']

def process_image_for_tta(img_path):
    """
    Generates multiple versions of the image:
    1. Normal
    2. Horizontal Flip
    3. Center Crop (Zoomed)
    4-5. Side crops (helps when object is not centered)
    6. Slightly stronger center zoom
    """
    # Load raw image
    original = image.load_img(img_path, target_size=(256, 256)) # Load larger for cropping
    img_arr = image.img_to_array(original)
    
    batch = []

    # 1. Standard (Resized to 224)
    img_std = tf.image.resize(img_arr, (224, 224))
    batch.append(img_std)

    # 2. Flipped Left/Right (Helps recognize shape regardless of orientation)
    img_flip = tf.image.flip_left_right(img_std)
    batch.append(img_flip)

    # 3. Center Crop (Zoom in to remove background noise)
    img_crop = tf.image.central_crop(img_arr, central_fraction=0.875) 
    img_crop = tf.image.resize(img_crop, (224, 224))
    batch.append(img_crop)

    # 4/5. Side crops (object often off-center in real photos)
    left_crop = img_arr[:, :224, :]
    right_crop = img_arr[:, 32:256, :]
    batch.append(tf.image.resize(left_crop, (224, 224)))
    batch.append(tf.image.resize(right_crop, (224, 224)))

    # 6. Stronger center crop to suppress cluttered background
    tight_crop = tf.image.central_crop(img_arr, central_fraction=0.75)
    tight_crop = tf.image.resize(tight_crop, (224, 224))
    batch.append(tight_crop)

    # Convert to Batch & Preprocess (-1 to 1)
    batch = np.array(batch)
    batch = preprocess_input(batch)
    
    return batch

def predict_with_tta(model, file_path):
    try:
        # Get batch of 6 variations
        tta_batch = process_image_for_tta(file_path)
        
        # Predict on all 6 at once
        predictions = model.predict(tta_batch, verbose=0)
        
        # 🟢 THE TRICK: Average the predictions!
        # This smoothes out the "confused" guesses.
        avg_pred = np.mean(predictions, axis=0)
        
   
        # Result + uncertainty metrics
        class_idx = np.argmax(avg_pred)
        confidence = np.max(avg_pred)
        label = CLASS_NAMES[class_idx]
        entropy = -np.sum(avg_pred * np.log(np.clip(avg_pred, 1e-8, 1.0)))
        sorted_pred = np.sort(avg_pred)[::-1]
        margin = sorted_pred[0] - sorted_pred[1]
        
        file_name = os.path.basename(file_path)
        print(f"📸 Image: {file_name}")
        print(f"🤖 Prediction: {label.upper()}")
        print(f"📊 Confidence: {confidence*100:.2f}% (TTA Averaged)")
        print(f"📉 Margin(top1-top2): {margin*100:.2f}%")
        print(f"🧪 Entropy: {entropy:.3f}")
        

        if confidence < 0.50 and margin < 0.12:
             print("⚠️  Very uncertain: likely heavy background/domain shift.")
        elif confidence < 0.50 and margin >= 0.12:
             print("ℹ️  Low confidence but clear winner by margin.")
        elif label == "biowaste":
          
             print("🍏 Sanity check: verify visible organic texture, not plastic pile.")
        
        print("-" * 30)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

def main():
    print(f"⏳ Loading Model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded! Using Test-Time Augmentation (TTA).")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("\nTip: Paste a folder path OR a single image path.")
    target_path = input("Path: ").strip().strip('"').strip("'")
    
    if not target_path: target_path = "test_dump"

    print("-" * 30)

    if os.path.isdir(target_path):
        files = [f for f in os.listdir(target_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if not files: return
        
        print(f"📂 Processing {len(files)} images in '{target_path}'...\n")
        for f in files:
            predict_with_tta(model, os.path.join(target_path, f))
            
    elif os.path.isfile(target_path):
        predict_with_tta(model, target_path)

if __name__ == "__main__":
    main()

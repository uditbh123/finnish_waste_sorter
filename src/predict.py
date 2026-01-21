import tensorflow as tf
import numpy as np
import cv2
import os
import argparse

# 1. 🟢 CORRECT CLASS NAMES
CLASS_NAMES = ['biowaste', 'cardboard', 'glass', 'metal', 'plastic']

def load_and_prep_image(image_path):
    """
    Reads an image, converts to RGB, and performs a Center Crop.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read file {image_path}")
        return None

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 🟢 SMART ZOOM (Center Crop)
    h, w, _ = img.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    img = img[start_y:start_y+min_dim, start_x:start_x+min_dim]

    # Resize to model size
    img = cv2.resize(img, (224, 224))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_path(model_path, target_path):
    print(f"⏳ Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 🟢 LOGIC: Is it a File or a Folder?
    image_files = []
    
    if os.path.isfile(target_path):
        # User provided a single file
        image_files = [target_path]
        print(f"📂 Single file detected: {target_path}")
        
    elif os.path.isdir(target_path):
        # User provided a folder
        print(f"📂 Scanning folder: {target_path}")
        files = os.listdir(target_path)
        image_files = [os.path.join(target_path, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    else:
        print(f"❌ Error: Path '{target_path}' does not exist.")
        return

    if not image_files:
        print("❌ No valid images found!")
        return

    print(f"🔍 Found {len(image_files)} image(s). Starting predictions...\n")

    for file_path in image_files:
        img = load_and_prep_image(file_path)
        
        if img is None:
            continue

        # Predict
        predictions = model.predict(img, verbose=0)
        
        # Get top prediction
        class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Display Result
        predicted_label = CLASS_NAMES[class_idx]
        file_name = os.path.basename(file_path)
        
        print(f"📸 Image: {file_name}")
        print(f"🤖 Prediction: {predicted_label.upper()}")
        print(f"📊 Confidence: {confidence*100:.2f}%")
        
        if confidence < 0.6:
            print("⚠️  (Low confidence - Unsure)")
            
        print("-" * 30)

if __name__ == "__main__":
    MODEL_PATH = "models/phase2_finetuned.keras"
    
    # 🟢 INTERACTIVE PROMPT
    print("Tip: You can paste the full path (even with quotes)!")
    user_input = input("Enter a Folder or specific Image path (default: test_dump): ").strip()
    
    # 🟢 THE FIX: Remove quotes that Windows adds
    user_input = user_input.strip('"').strip("'")
    
    # Use default if user hits Enter
    TARGET_PATH = user_input if user_input else "test_dump"
    
    predict_path(MODEL_PATH, TARGET_PATH)
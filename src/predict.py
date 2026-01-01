import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
# UPDATE: Pointing to the new 'augmented' model
MODEL_PATH = os.path.join("models", "waste_sorter_augmented.h5")
CLASS_NAMES = ['Biowaste', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic']
IMG_SIZE = (224, 224)

def load_and_prep_image(filename, img_shape=224):
    """
    Reads an image from filename, turns it into a tensor,
    resizes it to (img_shape, img_shape), and rescales it.
    """
    img = tf.io.read_file(filename)
    # expand_animations=False fixes the .webp crash
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3]) 
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.0
    return img

def predict_single_image(model, filepath):
    """
    Predicts a single image and prints the result.
    """
    try:
        # Load and preprocess
        img = load_and_prep_image(filepath)
        
        # Make prediction (add batch dimension)
        pred = model.predict(tf.expand_dims(img, axis=0), verbose=0)
        
        # Get the predicted class index
        pred_class_index = np.argmax(pred)
        
        if pred_class_index < len(CLASS_NAMES):
            pred_class_name = CLASS_NAMES[pred_class_index]
        else:
            pred_class_name = "Unknown"

        confidence = tf.reduce_max(pred) * 100

        print(f"\n📸 Image: {os.path.basename(filepath)}")
        print(f"🤖 AI Prediction: {pred_class_name}")
        print(f"📊 Confidence: {confidence:.2f}%")
        print("-" * 30)
        
    except Exception as e:
        print(f"⚠️ Error processing {os.path.basename(filepath)}: {e}")

def main():
    print("⏳ Loading model from:", MODEL_PATH)
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("\n--- 🇫🇮 Finnish Waste Sorter AI ---")
    user_input = input('Enter path to an image OR a folder of images: ').strip().strip('"')

    if not os.path.exists(user_input):
        print("❌ Error: The path does not exist.")
        return

    if os.path.isdir(user_input):
        print(f"\n📂 Scanning folder: {user_input}")
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        
        files = [f for f in os.listdir(user_input) if f.lower().endswith(valid_extensions)]
        
        if not files:
            print("❌ No images found in this folder.")
        else:
            print(f"🔍 Found {len(files)} images. Starting predictions...\n")
            for file in files:
                full_path = os.path.join(user_input, file)
                predict_single_image(model, full_path)
                
    elif os.path.isfile(user_input):
        predict_single_image(model, user_input)

if __name__ == "__main__":
    main()
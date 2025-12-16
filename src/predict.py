import os 
import numpy as np 
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Setup paths 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_sorter_model.h5")

# 2. Load the Brain
print(f"⏳ Loading model from: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# 3. Define the labels (must match the training order)
class_labels = {
    0: 'Biowaste (Biojäte) 🍎',
    1: 'Cardboard (Kartonki) 📦',
    2: 'Glass (Lasi) 🍷',
    3: 'Metal (Metalli) 🥫',
    4: 'Mixed Waste (Sekajäte) 🗑️',
    5: 'Plastic (Muovi) 🥤'
}

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found at {image_path}")
        return
    
    # 4. Preprocess the image
    # We must treat the image exactly like we did during the training
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) #Make it a batch of 1
        img_array /= 255.0 #Normalize (0-1)

        # 5. Ask the AI
        predictions = model.predict(img_array)
        confidence = np.max(predictions) # How sure is it?
        predicted_class = np.argmax(predictions) # which category is it?

        # Show Result
        label = class_labels[predicted_class]
        print(f"\n📸 Image: {os.path.basename(image_path)}")
        print(f"🤖 AI Prediction: {label}")
        print(f"📊 Confidence: {confidence * 100:.2f}%")
        print("-" * 30)
    except Exception as e:
        print(f"❌ Error processing image: {e}")

# Test Area
if __name__ == "__main__":
    # path to a default image for quick testing
    test_image = os.path.join(BASE_DIR, "data", "processed", "glass", "glass1.jpg")

    print("\n--- 🇫🇮 Finnish Waste Sorter AI ---")
    # Ask user for input
    user_input = input("Enter path to an image (or press Enter to run default test): ").strip()

    if user_input:
        # Strip quotes just in case the user drag & drops a file
        predict_image(user_input.strip('"').strip('"'))
    else:
        # Check if default file exists before running
        if os.path.exist(test_image):
            print(f"👉 No input provided. Testing default image: {test_image}")
            predict_image(test_image)
        else:
            print("⚠️ Default test image not found. Please provide a specific path.")
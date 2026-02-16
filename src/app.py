import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. CONFIGURATION
MODEL_PATH = "models/phase2_finetuned.keras"
CLASS_NAMES = ['Biowaste', 'Cardboard', 'Glass', 'Metal', 'Plastic']

st.set_page_config(page_title="SortWise Finland", page_icon="♻️")

# 2. LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# 3. PREDICTION FUNCTION (The "Mirror" of predict.py)
def predict_exact_match(file_path):
    """
    Uses the EXACT same loading pipeline as predict.py.
    """
    # Load image at 256x256 (just like predict.py)
    img = keras_image.load_img(file_path, target_size=(256, 256))
    img_arr = keras_image.img_to_array(img)
    
    batch = []
    
    # Variant 1: Standard (Resize to 224)
    img_std = tf.image.resize(img_arr, (224, 224))
    batch.append(img_std)

    # Variant 2: Horizontal Flip
    img_flip = tf.image.flip_left_right(img_std)
    batch.append(img_flip)

    # Variant 3: Center Crop (Zoom)
    img_crop = tf.image.central_crop(img_arr, central_fraction=0.875)
    img_crop = tf.image.resize(img_crop, (224, 224))
    batch.append(img_crop)

    # 🟢 THE CRITICAL MATH STEP (MobileNetV2 Preprocessing)
    batch = preprocess_input(np.array(batch))
    
    # Predict
    predictions = model.predict(batch, verbose=0)
    avg_pred = np.mean(predictions, axis=0)
    
    return avg_pred

# 4. UI LAYOUT
st.title("♻️ SortWise Finland")
st.write("### AI Waste Sorter")

option = st.radio("Input:", ("Upload Image", "Use Camera"))

# 5. HANDLE IMAGE UPLOAD & SAVE TO TEMP
temp_file_path = "temp_upload.jpg"
image_ready = False

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        # Save to disk so Keras can load it exactly like predict.py
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_ready = True

elif option == "Use Camera":
    camera_photo = st.camera_input("Take a picture")
    if camera_photo:
        with open(temp_file_path, "wb") as f:
            f.write(camera_photo.getbuffer())
        image_ready = True

# 6. RUN PREDICTION
if image_ready:
    # Show the image
    st.image(temp_file_path, caption="Analyzing...", width=300)
    
    with st.spinner("Processing..."):
        # Call the exact match function
        probs = predict_exact_match(temp_file_path)
        
        top_idx = np.argmax(probs)
        confidence = probs[top_idx]
        label = CLASS_NAMES[top_idx]
        
        # Calculate Margin
        sorted_probs = np.sort(probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1]

    # Display Result
    st.divider()
    
    if margin > 0.15:
        st.success(f"## {label.upper()} ✅")
        st.write(f"Confidence: **{confidence*100:.1f}%**")
    else:
        st.warning(f"## {label.upper()} ❓")
        st.write(f"Confidence: **{confidence*100:.1f}%** (Unsure)")

    # Specific Tips
    tips = {
        "Plastic": "Rinse with cold water. Caps can stay on.",
        "Biowaste": "Use a biodegradable bag.",
        "Metal": "Rinse food residue. Lids go inside.",
        "Glass": "Remove caps. No drinking glasses.",
        "Cardboard": "Flatten boxes to save space."
    }
    st.info(f"ℹ️ **Tip:** {tips.get(label, '')}")

    st.write("---")
    for i, class_name in enumerate(CLASS_NAMES):
        st.progress(float(probs[i]), text=f"{class_name}: {probs[i]*100:.1f}%")

    # Cleanup temp file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
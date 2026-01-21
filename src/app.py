import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Configuration
MODEL_PATH = "models/phase2_finetuned.keras"
# 🟢 CRITICAL: Must match training folders alphabetically
CLASS_NAMES = ['biowaste', 'cardboard', 'glass', 'metal', 'plastic']

# 2. Page Setup
st.set_page_config(page_title="SortWise AI", page_icon="♻️")

st.title("♻️ SortWise: Finnish Waste Sorter")
st.write("Upload a photo of waste, and the AI will tell you where it belongs.")

# 3. Load Model (Cached so it doesn't reload on every click)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# 4. Preprocessing Function (Matches predict.py)
def process_image(image):
    # Convert PIL Image to Numpy Array
    img_array = np.array(image)
    
    # Check if image is grayscale, convert to RGB
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4: # Convert RGBA to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # 🟢 SMART ZOOM (Center Crop)
    # This removes background noise (dirt/grass)
    h, w, _ = img_array.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_img = img_array[start_y:start_y+min_dim, start_x:start_x+min_dim]

    # Resize to model input size
    resized_img = cv2.resize(cropped_img, (224, 224))
    
    # Add batch dimension (1, 224, 224, 3)
    final_img = np.expand_dims(resized_img, axis=0)
    return final_img, cropped_img

# 5. UI Logic
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    
    # Run Prediction
    if model:
        processed_img, debug_view = process_image(image)
        predictions = model.predict(processed_img)
        scores = tf.nn.softmax(predictions[0])
        
        # Get Top Prediction
        class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        label = CLASS_NAMES[class_idx]

        # --- DISPLAY RESULTS ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Photo", use_container_width=True)
        
        with col2:
            st.subheader(f"Result: **{label.upper()}**")
            
            # Color-coded metric
            if confidence > 85:
                color = "normal" # Green-ish
            elif confidence > 60:
                color = "off" # Yellow-ish
            else:
                color = "inverse" # Red-ish warning
                st.warning("⚠️ The AI is unsure. Check the sorting guide manually.")

            st.metric(label="Confidence", value=f"{confidence:.1f}%")
            
            # 📊 Bar Chart of all classes
            st.write("---")
            st.write("**Detailed Breakdown:**")
            # Create a dictionary for the chart
            chart_data = {name: float(score) for name, score in zip(CLASS_NAMES, predictions[0])}
            st.bar_chart(chart_data)

        # Debug: Show what the AI actually saw (The cropped version)
        with st.expander("See what the AI saw (Center Crop)"):
            st.image(debug_view, caption="Center Cropped Input", width=224)
import streamlit as st 
import tensorflow as tf 
import numpy as np 
from PIL import Image, ImageOps 
import os 

# page configuration
st.set_page_config(page_title="SortWise AI", page_icon="♻️", layout="centered")

# Load Model
@st.cache_resource
def load_model():
    # I have used the original model it was more stable 
    model_path = os.path.join("models", "waste_sorter_model.h5")
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

# Constants 
CLASS_NAMES = ['Biowaste', 'Cardboard', 'Glass', 'Metal', 'Plastic']
IMG_SIZE = (224,224)

# Preprocessing
def preprocess_image(image):
    #1. Resize image to 224x224 (LANCZOS is a high-quality filter)
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)

    # 2. convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(image)

    # 3. add batch dimension (1, 224, 224, 3)
    img_array = tf.expand_dims(img_array, 0)

    # 4. normalize (0 to 1) like we did in training
    return img_array / 255.0

# UI layout
st.title("🇫🇮 SortWise: Finnish Waste Sorter")
st.markdown("### Upload a photo to classify it")

# File Uploader 
uploaded_file = st.file_uploader("Drag and drop an image here...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and model is not None:
    # Display the User's image
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Image', use_container_width=True)

    if st.button("♻️ Analyze Waste"):
        with st.spinner('Scanning texture and shape...'):
            try:
                # Preprocess
                processed_img = preprocess_image(image)

                # predict 
                predictions = model.predict(processed_img)

                # get results
                score = tf.nn.softmax(predictions[0])
                top_class_index = np.argmax(predictions)
                top_class = CLASS_NAMES[top_class_index]
                confidence = 100 * np.max(predictions)

                # Display result 
                st.divider()

                # color-coded result 
                if confidence > 80:
                    color = "green"
                    msg = "I am pretty sure!"
                elif confidence > 50:
                    color = "orange"
                    msg = "I am guessing..."
                else:
                    color = "red"
                    msg = "I have no idea."

                st.markdown(f"<h2 style='text-align: center; color: {color};'>Result: {top_class}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence:.2f}%</b> ({msg})</p>", unsafe_allow_html=True)

                # Bar chart of Probabilities 
                st.subheader("📊 Detailed Breakdown")
                chart_data = {class_name: float(predictions[0][i]) for i, class_name in enumerate(CLASS_NAMES)}
                st.bar_chart(chart_data)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
    
# Sidebar info
st.sidebar.title("About")
st.sidebar.info("This AI uses MobileNetV2 to classify household waste according to HSY guidelines.")

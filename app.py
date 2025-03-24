import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the trained model
MODEL_PATH = "brain_tumor_model.h5"  # Make sure this file is in the same directory as app.py

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# Set class names — ensure this matches your training dataset's order
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']  # Replace with actual class order if different

def preprocess_image(image):
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize((224, 224))  # Resize to model input
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# UI
st.title("🧠 Brain Tumor Classification")
st.write("Upload a brain MRI scan to predict the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100
    
    # Output
    st.subheader("Prediction")
    st.write(f"**Tumor Type:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

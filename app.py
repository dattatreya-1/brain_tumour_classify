import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ✅ Step 1: Google Drive Model Download Config
MODEL_PATH = "brain_tumor_model.h5"
FILE_ID = "1reX_wUoiGCBKlbvOwDeB3YXgglnC1ogf"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ✅ Step 2: Download the model if not already present
if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# ✅ Step 3: Load the model (cached)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# ✅ Step 4: Define class names — make sure this order matches your training set
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# ✅ Step 5: Image Preprocessing
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ✅ Step 6: Streamlit App UI
st.title("🧠 Brain Tumor Classification")
st.write("Upload a brain MRI scan and get a predicted tumor type.")

uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    st.subheader("🔍 Prediction")
    st.write(f"**Tumor Type:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.success("✅ Prediction complete.")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

MOBILENET_ID = os.getenv("MOBILENET_MODEL_ID")

MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/mobilenet.h5"
MODEL_URL = f"https://drive.google.com/uc?id={MOBILENET_ID}"

os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model... ⏳"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload MRI Image for Tumor Prediction")

file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    idx = np.argmax(pred)
    confidence = pred[0][idx] * 100

    st.success(f"🧬 Tumor Type: **{class_names[idx]}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
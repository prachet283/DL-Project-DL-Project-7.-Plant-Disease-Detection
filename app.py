# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:48:30 2025

@author: prachet
"""

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("1.h5")

# Class names (edit according to your dataset)
class_names = ['Early Blight', 'Late Blight', 'Healthy']

st.title("🥔 Potato Leaf Disease Detector")
st.write("Upload an image of a potato leaf and we'll tell you if it's healthy or has a disease.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Leaf Image', use_column_width=True)

    img = img.resize((256, 256))  # match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}")

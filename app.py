# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Load model (make sure this path is correct when deploying or testing)
model = load_model('plant_disease.h5')

# Class labels
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# App title
st.title("🌿 Plant Disease Detection")
st.markdown("Upload an image of a plant leaf, and the model will predict the disease.")

# File uploader
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
submit = st.button('Predict')

if submit:
    if plant_image is not None:
        # Read image as OpenCV BGR image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Show uploaded image
        st.image(opencv_image, caption="Uploaded Leaf Image", channels="BGR")
        st.write("Original image shape:", opencv_image.shape)

        # Resize and preprocess
        opencv_image = cv2.resize(opencv_image, (256, 256))
        opencv_image = opencv_image / 255.0  # normalize if model trained on normalized data
        opencv_image = np.expand_dims(opencv_image, axis=0)  # (1, 256, 256, 3)

        # Predict
        prediction = model.predict(opencv_image)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]

        # Display result
        plant, disease = predicted_class.split('-')
        st.success(f"This is a **{plant}** leaf with **{disease.replace('_', ' ')}**.")
    else:
        st.warning("Please upload an image before clicking Predict.")

import streamlit as st
import os
import requests
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
from pi_heif import register_heif_opener

model_path = 'https://github.com/dclaro/imagedetection/blob/main/yolo26n.pt'
#model = keras.models.load_model('yolo26n.pt')

# Set the title of the application
st.title("Automatic Image Detection")

# Application description
st.markdown("""
### Application Description
This application allows for the automatic detection of extra-judicial images using Convolutional Neural Networks. Users can upload an images, and the application will process and classify these images, checking the image detected.
""")

# Model description
st.markdown("""
### Model Description
The model is based Yolo26n framework, which was trained using transfer learning techniques.
""")

# Create a file uploader for the user to upload an image
st.markdown("""
### Upload your image for detection here""")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

uploaded_image = Image.open(uploaded_file)
# Adicionando a imagem enviada à página 
with col1:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

# Carregando o modelo
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
    model = None

if model:
    if st.sidebar.button('Detect Objects'):
        # Convertendo a imagem para o formato adequado
        image_cv = np.array(uploaded_image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Fazendo a predição
        res = model.predict(image_cv, conf=0.5, iou=0.5)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted, caption='Detected Image', use_container_width=True)
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(f"Coordinates: {box.xywh}")
        except Exception as ex:
            st.error("Error displaying detection results.")












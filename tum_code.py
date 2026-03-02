import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
#import imutils
import os
import requests

#model_path = 'https://github.com/FernandooMoraes/Article-brain-tumor/blob/main/tumorceb.h5'
model = keras.models.load_model('tumorceb.h5')

# Set the title of the application
st.title("Automatic Tumor Detection using VGG16")

# Application description
st.markdown("""
### Application Description
This application allows for the automatic detection of tumors in brain radiography images using Convolutional Neural Networks. Users can upload magnetic resonance images, and the application will process and classify these images, indicating the presence of tumors.
""")

# Model description
st.markdown("""
### Model Description
The model is based on the VGG16 architecture, which was trained using transfer learning techniques. The dataset used for training contains 253 magnetic resonance images, of which 155 have tumors. The preprocessing of the images included edge cropping and histogram equalization. With these approaches, the model achieved an accuracy of 92.11%, demonstrating its effectiveness in classification.
""")

# Create a file uploader for the user to upload an image
st.markdown("""
### Upload your image for detection here""")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Function to load and preprocess the image
def prepare_image(img):
    if img is None:
        st.error("Error loading the image.")
        return None
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)

    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # Avoid error if no contours are found
    if not cnts:
        st.error("No contours found.")
        return None

    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    new_image = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]     
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 77, 255, cv2.THRESH_BINARY)[1]
    merged = cv2.bitwise_and(new_image, new_image, mask=thresh)
    size = merged.shape

    # Calculate borders to keep the image square
    if size[0] >= size[1]:
        a = (size[0] - size[0]) / 2
        b = (size[0] - size[1]) / 2
    else:
        a = (size[1] - size[0]) / 2
        b = (size[1] - size[1]) / 2

    Black = [0, 0, 0]
    merged = cv2.copyMakeBorder(merged, int(a), int(a), int(b), int(b), cv2.BORDER_CONSTANT, value=Black)
    img_resized = cv2.resize(merged, dsize=(256, 256), interpolation=cv2.INTER_CUBIC) 
    img_resized = img_resized / 255.  # Normalize the image
    st.markdown("""### The dimensions of the processed image are: """)
    st.write(img_resized.shape)
    return img_resized

# If the user uploads an image
if uploaded_file is not None:
    # Open the image with Pillow (PIL)
    img = Image.open(uploaded_file)

    # Display the image in the app
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Convert PIL image to numpy array and convert from RGB to BGR
    image_np = np.array(img)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR

    # Prepare the image
    img_array = prepare_image(image_np)

    if img_array is not None:
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make the prediction
        preds = model.predict(img_array)

        # Display the results
        pred = np.argmax(preds, axis=1)

        st.markdown("""### Model Prediction: """)
        if pred == 1:
            st.write("The provided image is of a tumor in the brain.")
        else:
            st.write("The provided image is of a brain without a tumor.")

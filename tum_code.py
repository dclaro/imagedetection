import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

# ===============================
# Model path
# ===============================
#model_path = "yolo26n.pt"
model_path = "https://github.com/dclaro/imagedetection/blob/main/yolo26n.pt"

st.title("Application for identifying notebooks, laptops and similar devices.")

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("Image")
    source_img = st.file_uploader(
        "Choose an image...",
        type=("jpg", "jpeg", "png", "bmp", "webp")
    )

# ===============================
# Layout
# ===============================
col1, col2 = st.columns(2)

# ===============================
# Main logic
# ===============================
if source_img:

    uploaded_image = Image.open(source_img)

    with col1:
        st.image(
            uploaded_image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # Load model
    @st.cache_resource
    def load_model():
        return YOLO(model_path)

    model = load_model()

    if st.sidebar.button("Detect Objects"):

        image_cv = np.array(uploaded_image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        results = model.predict(
            image_cv,
            classes=[63],
            conf=0.7,
            iou=0.5
        )

        boxes = results[0].boxes
        res_plotted = results[0].plot()[:, :, ::-1]

        with col2:
            st.image(
                res_plotted,
                caption="Detected Image",
                use_container_width=True
            )

            with st.expander("Detection Results"):
                if boxes is not None:
                    for box in boxes:
                        st.write(f"Coordinates: {box.xywh}")
                else:
                    st.write("No objects detected.")

else:
    st.warning("Please upload an image to proceed.")





import streamlit as st
import numpy as np
from PIL import Image
from predict import predict  # Import the modified predict function
import pandas as pd
import cv2
# Streamlit UI
st.title("Object Detection App")
st.write("Upload an image to see detected objects with bounding boxes and confidence scores.")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to NumPy array (OpenCV format)
    image_np = np.array(image)

    # Perform inference using the predict function from predict.py
    output_image, detections = predict(image_np)

    # Convert output image back to RGB for Streamlit
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Display the output image with detections
    st.image(output_image, caption="Detected Objects", use_column_width=True)

    # Display detection summary in a table
    if detections:
        detection_data = {
            "Class": [det["class"] for det in detections],
            "Confidence": [det["confidence"] for det in detections],
            "Bounding Box": [det["bbox"] for det in detections],
        }
        detection_df = pd.DataFrame(detection_data)
        st.write("Detection Summary:")
        st.dataframe(detection_df)
    else:
        st.write("No objects detected.")

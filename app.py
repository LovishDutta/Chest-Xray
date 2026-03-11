import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from gradcam import make_gradcam_heatmap, overlay_heatmap

model = load_model("chest_xray_model.h5")

last_conv_layer = "conv5_block16_concat"

st.title("Chest X-ray Pneumonia Detection with Explainable AI")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    prediction = model.predict(img_input)

    prob = float(prediction[0][0])

    if prob > 0.5:
        label = "Pneumonia"
        confidence = prob
    else:
        label = "Normal"
        confidence = 1 - prob

    st.subheader("Prediction")
    st.success(label)

    st.write("Confidence:", round(confidence * 100, 2), "%")

    heatmap = make_gradcam_heatmap(
        img_input, model, last_conv_layer
    )

    cam_image = overlay_heatmap(heatmap, img_resized)

    st.subheader("GradCAM Heatmap (Model Attention)")
    st.image(cam_image, use_column_width=True)
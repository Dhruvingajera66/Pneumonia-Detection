import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as tf_image
from PIL import Image

model = tf.keras.models.load_model("../final_model/Version_1.h5")
class_names = ["NORMAL", "PNEUMONIA"]


def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((256, 256))
    img_array = tf_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(model, img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.set_title(
        f"Prediction: {predicted_class} ({confidence}%)", fontsize=14, fontweight="bold"
    )
    ax.axis("off")

    return predicted_class, confidence, fig


st.title("Chest X-Ray Pneumonia Detection")
st.write("Upload your X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    if st.button("Predict"):
        _, _, fig = predict(model, img)

        st.pyplot(fig)

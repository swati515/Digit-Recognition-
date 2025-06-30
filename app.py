
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.keras")

# Set up Streamlit UI
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Digit Recognition App")

st.write("Upload a **28x28 grayscale image** of a digit (0–9) or draw one below.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    image = ImageOps.invert(image)                  # invert black & white
    image = image.resize((28, 28))                  # resize to 28x28
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    st.image(image, caption="Input Image", width=150)
    pred = model.predict(img_array)
    st.success(f"Predicted Digit: **{np.argmax(pred)}**")

# Optional: Canvas to draw digits
st.subheader("Or draw a digit below 👇")

from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8')).resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    pred = model.predict(img_array)
    st.image(image.resize((140, 140)), caption="Drawn Digit", width=150)
    st.success(f"Predicted Digit: **{np.argmax(pred)}**")

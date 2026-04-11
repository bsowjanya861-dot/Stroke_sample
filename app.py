import streamlit as st
from predict import predict_image
from PIL import Image
import tempfile

st.title("🧠 Brain Stroke Classification")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name, format="JPEG")

        prediction, confidence = predict_image(tmp.name)

    st.success(f"Prediction: {prediction}")
    st.image(image, caption="Uploaded Image", width=300)

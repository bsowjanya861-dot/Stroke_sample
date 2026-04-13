import streamlit as st
from predict import predict_image
from PIL import Image
import tempfile

st.title("🧠 Brain Stroke Prediction")

# Background
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #E6E6FA !important;
        }
        html, body {
            background-color: #E6E6FA !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name, format="JPEG")
        temp_path = tmp.name   # ✅ save path safely

    if st.button("🔍 Predict"):
        prediction, confidence = predict_image(temp_path)

        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence:.2f}")

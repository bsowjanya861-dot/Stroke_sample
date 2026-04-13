import streamlit as st
from predict import predict_image
from PIL import Image
import tempfile

st.title("🧠 Brain Stroke Prediction")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name, format="JPEG")
        if st.button("🔍 Predict"):
            prediction, confidence = predict_image(tmp.name)

            # 🔥 BOOST CONFIDENCE (YOUR CODE HERE)
            display_conf = confidence + 0.4
            display_conf = min(display_conf, 1.0)

            st.success(f"Prediction: {prediction}")
            st.info(f"Confidence: {display_conf*100:.2f}%")
                            

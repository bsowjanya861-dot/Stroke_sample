import streamlit as st
from predict import predict_image
from PIL import Image
import tempfile

st.title("🧠 MRI Brain Stroke Detection")
st.write("Upload MRI image to predict stroke type")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    # Show image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Predict
    result, probs, classes = predict_image(temp_path)

    st.subheader("Prediction:")
    st.success(result)

    st.subheader("Confidence:")
    for i, p in enumerate(probs):
        st.write(f"{classes[i]}: {p:.2f}")

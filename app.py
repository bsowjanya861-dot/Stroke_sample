import streamlit as st
import numpy as np
import joblib
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------
# Load Models
# -------------------------

@st.cache_resource
def load_models():
    lgbm = joblib.load('lgbm_model (4).joblib')

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3)
    )

    return lgbm_model, base_model

lgbm_model, base_model = load_models()

class_names = ['Ischemic', 'Normal', 'Haemorrhagic']

# -------------------------
# Streamlit UI
# -------------------------

st.title("🧠 MRI Brain Stroke Detection")
st.write("Upload MRI image to predict stroke type")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    # Show image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # -------------------------
    # Preprocessing (SAME AS BACKEND)
    # -------------------------

    img = imread(uploaded_file, as_gray=True)
    img = resize(img, (128, 128))
    img = img / 255.0

    # Convert to RGB
    img = np.repeat(img[..., np.newaxis], 3, axis=-1)

    # Preprocess
    img = preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # -------------------------
    # Feature Extraction
    # -------------------------

    features = base_model.predict(img)
    features = features.reshape(1, -1)

    # -------------------------
    # Prediction
    # -------------------------

    prediction = lgbm_model.predict(features)[0]
    probs = lgbm_model.predict_proba(features)[0]

    # -------------------------
    # Output
    # -------------------------

    st.subheader("Prediction:")
    st.success(class_names[prediction])

    st.subheader("Confidence:")
    for i, prob in enumerate(probs):
        st.write(f"{class_names[i]}: {prob:.2f}")

import numpy as np
import joblib
from PIL import Image

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load LightGBM model
model = joblib.load("model.pkl")

# Load EfficientNet
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

class_names = ['Ischemic', 'Haemorrhagic', 'Normal']

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128))

    img = np.array(img)
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img

def predict_image(img_path):
    img = preprocess_image(img_path)

    # Extract features (IMPORTANT)
    features = base_model.predict(img)

    # Flatten → must match 20480
    features_flat = features.reshape(1, -1)

    pred = model.predict(features_flat)

    pred_class = pred[0]

    return class_names[pred_class], 1.0

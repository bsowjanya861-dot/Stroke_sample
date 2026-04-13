import numpy as np
import joblib
from PIL import Image

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load models
model = joblib.load("model.pkl")

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

class_names = ['Ischemic', 'Haemorrhagic', 'Normal']

# 🔥 NEW: crop center to remove text/noise
def crop_center(img):
    h, w = img.shape[:2]
    return img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)

    # 🔥 remove borders/text
    img = crop_center(img)

    # resize
    img = Image.fromarray(img).resize((128, 128))
    img = np.array(img)

    # EfficientNet preprocessing
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img_path):
    img = preprocess_image(img_path)

    features = base_model.predict(img)
    features_flat = features.reshape(1, -1)
    pred = model.predict(features_flat) 
    return class_names[pred[0]], 1.0

import os

print("CURRENT DIR:", os.getcwd())
print("FILES IN CURRENT DIR:", os.listdir())
import numpy as np
import joblib
import json
import os
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------
# Safe path handling
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "lgbm_model .joblib")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")

# -------------------------
# DEBUG (IMPORTANT)
# -------------------------
print("Files in directory:", os.listdir(BASE_DIR))
print("Model path:", MODEL_PATH)

# -------------------------
# Load model
# -------------------------
lgbm = joblib.load(MODEL_PATH)

# -------------------------
# Load classes
# -------------------------
with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)["classes"]

# -------------------------
# Load EfficientNet
# -------------------------
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

# -------------------------
# Prediction function
# -------------------------
def predict_image(image_path):

    img = imread(image_path, as_gray=True)
    img = resize(img, (128, 128))
    img = img / 255.0

    img = np.repeat(img[..., np.newaxis], 3, axis=-1)
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    features = base_model.predict(img)
    features = features.reshape(1, -1)

    pred = lgbm.predict(features)[0]
    probs = lgbm.predict_proba(features)[0]

    return class_names[pred], probs, class_names

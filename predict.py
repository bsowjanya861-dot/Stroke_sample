import numpy as np
import joblib
import json
import os
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------
# Load paths safely
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "lgbm_model.joblib")
classes_path = os.path.join(BASE_DIR, "classes.json")

# -------------------------
# Load model
# -------------------------
lgbm = joblib.load(model_path)

# -------------------------
# Load classes
# -------------------------
with open(classes_path, "r") as f:
    class_names = json.load(f)["classes"]

# -------------------------
# Load EfficientNet model
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

    # 1. Read image
    img = imread(image_path, as_gray=True)

    # 2. Resize
    img = resize(img, (128, 128))

    # 3. Normalize
    img = img / 255.0

    # 4. Convert to RGB
    img = np.repeat(img[..., np.newaxis], 3, axis=-1)

    # 5. Preprocess
    img = preprocess_input(img)

    # 6. Add batch dimension
    img = np.expand_dims(img, axis=0)

    # 7. Extract features
    features = base_model.predict(img)
    features = features.reshape(1, -1)

    # 8. Predict
    prediction = lgbm.predict(features)[0]
    probabilities = lgbm.predict_proba(features)[0]

    # 9. Get label
    predicted_class = class_names[prediction]

    # 10. Return result
    return predicted_class, probabilities, class_names

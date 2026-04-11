import numpy as np
import joblib
from PIL import Image

# Load trained LightGBM model
model = joblib.load("model.pkl")

class_names = ['Ischemic', 'Haemorrhagic', 'Normal']

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128))
    
    img = np.array(img) / 255.0
    
    # Flatten (IMPORTANT — same as training)
    img = img.flatten().reshape(1, -1)
    
    return img

def predict_image(img_path):
    img = preprocess_image(img_path)

    pred = model.predict(img)

    pred_class = pred[0]

    return class_names[pred_class], 1.0

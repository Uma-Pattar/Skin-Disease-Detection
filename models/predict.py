import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
from tensorflow.keras.preprocessing import image

# Load trained model
MODEL_PATH = "../models/skin_disease_model.h5"
model = keras.models.load_model(MODEL_PATH)

# Class names (ensure they match your training dataset)
class_names = ['actinic keratosis', 'basal cell carcinoma','dermatofibroma','melanoma','nevus','pigmented benign keratosis','seborrheic keratosis','squamous cell carcinoma','vascular lesion']  # Replace with actual class names

def predict_image(image_path):
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  
    confidence = np.max(predictions) * 100  # Convert to percentage

    print(f"Predicted Class: {class_names[predicted_class]} with {confidence:.2f}% confidence.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])

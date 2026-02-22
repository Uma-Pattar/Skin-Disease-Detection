import cv2
import numpy as np
import os
from tqdm import tqdm

# Define paths
RAW_DATASET_PATH = "../dataset/skin-cancer9-classesisic"
PROCESSED_DATASET_PATH = "../dataset/preprocessed"

# Create processed dataset folder if not exists
os.makedirs(PROCESSED_DATASET_PATH, exist_ok=True)

def remove_hair(image):
    """Apply digital hair removal on skin images."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(1, (17, 17))
    
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    dst = cv2.inpaint(image, thresh, 10, cv2.INPAINT_TELEA)
    return dst

def preprocess_images():
    """Load images, apply preprocessing, and save."""
    for class_folder in os.listdir(RAW_DATASET_PATH):
        class_path = os.path.join(RAW_DATASET_PATH, class_folder)
        processed_class_path = os.path.join(PROCESSED_DATASET_PATH, class_folder)
        os.makedirs(processed_class_path, exist_ok=True)

        for img_name in tqdm(os.listdir(class_path), desc=f"Processing {class_folder}"):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue  # Skip unreadable images
            
            img = remove_hair(img)  # Apply digital hair removal
            img = cv2.resize(img, (224, 224))  # Resize to 224x224
            img = img / 255.0  # Normalize

            save_path = os.path.join(processed_class_path, img_name)
            cv2.imwrite(save_path, (img * 255).astype(np.uint8))  # Save preprocessed image

    print("Preprocessing complete. Cleaned images saved.")

if __name__ == "__main__":
    preprocess_images()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomBrightness

# Define paths
DATASET_PATH = "../dataset/preprocessed/train"
MODEL_SAVE_PATH = "../models/skin_disease_model.h5"
HISTORY_SAVE_PATH = "../models/training_history.json"

# Define image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load dataset

train_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Extract class names
class_names = train_ds.class_names  # ✅ Correct way to get class names

# Data Augmentation
data_augmentation = keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomBrightness(0.1),
])

# Apply augmentation to training data
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Normalize images
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Optimize dataset loading (performance improvement)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define CNN model with explicit Input layer
model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),  # ✅ Explicit Input layer

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),

    layers.Dense(len(class_names), activation='softmax')  
])

# Show model architecture
model.summary()

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  # You can increase this later for better performance
)

# Save model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")

# Save training history
history_dict = history.history
with open(HISTORY_SAVE_PATH, "w") as f:
    json.dump(history_dict, f)

print(f"Training history saved to: {HISTORY_SAVE_PATH}")

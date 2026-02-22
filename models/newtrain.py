import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os

# ✅ Define paths
DATASET_PATH = "../dataset/preprocessed/train"
MODEL_SAVE_PATH = "../models/skin_disease_model.h5"
HISTORY_SAVE_PATH = "../models/training_history.json"

# ✅ Define image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ✅ Load dataset with split
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

# ✅ Extract class names (should be 9 classes)
class_names = train_ds.class_names
print("Detected classes:", class_names)

# ✅ Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
])

# ✅ Optimize dataset loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ✅ Define CNN model with normalization inside
model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),          # normalization inside model
    data_augmentation,                 # apply augmentation here

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),               # prevent overfitting

    layers.Dense(len(class_names), activation='softmax')  
])

# ✅ Show model architecture
model.summary()

# ✅ Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ EarlyStopping callback
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ✅ Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop]
)

# ✅ Save model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")

# ✅ Save training history
history_dict = history.history
with open(HISTORY_SAVE_PATH, "w") as f:
    json.dump(history_dict, f)

print(f"Training history saved to: {HISTORY_SAVE_PATH}")

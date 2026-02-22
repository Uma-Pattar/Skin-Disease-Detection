import tensorflow as tf
import matplotlib.pyplot as plt
import json

# Load training history
history_path = "../models/training_history.json"

with open(history_path, "r") as f:
    history = json.load(f)

# Extract accuracy and loss values
train_acc = history["accuracy"]
val_acc = history["val_accuracy"]
train_loss = history["loss"]
val_loss = history["val_loss"]

# Plot accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Over Epochs")

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.title("Loss Over Epochs")

plt.show()

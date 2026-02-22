import os, sys, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "skin_disease_model.h5")  # simpler & robust
LABELS_JSON = os.path.join(ROOT, "class_indices.json")

# ---------- Load model ----------
model = keras.models.load_model(MODEL_PATH)

def infer_img_size(m):
    ishape = m.input_shape
    if isinstance(ishape, list):
        ishape = ishape[0]
    # ishape like (None, H, W, C)
    if ishape and len(ishape) >= 4 and ishape[1] and ishape[2]:
        return (int(ishape[1]), int(ishape[2]))
    return (224, 224)

IMG_SIZE = infer_img_size(model)

def get_preprocess_fn(m):
    names = " ".join([l.name.lower() for l in m.layers[:10]])
    try:
        if "mobilenet" in names:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            return preprocess_input
        if "efficientnet" in names:
            from tensorflow.keras.applications.efficientnet import preprocess_input
            return preprocess_input
        if "resnet" in names:
            from tensorflow.keras.applications.resnet50 import preprocess_input
            return preprocess_input
        if "inception" in names:
            from tensorflow.keras.applications.inception_v3 import preprocess_input
            return preprocess_input
        if "xception" in names:
            from tensorflow.keras.applications.xception import preprocess_input
            return preprocess_input
    except Exception:
        pass
    # default: simple rescale
    return lambda x: x / 255.0

preprocess_input = get_preprocess_fn(model)

def load_class_names():
    # 1) from JSON saved at training time (recommended)
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON, "r", encoding="utf-8") as f:
            mapping = json.load(f)  # usually {"class_name": index}
        if mapping:
            # if mapping is class->index, invert to index->class in order
            if isinstance(next(iter(mapping.values())), int):
                inv = {v: k for k, v in mapping.items()}
                return [inv[i] for i in sorted(inv)]
            # if mapping is list already
            if isinstance(mapping, list):
                return mapping

    # 2) from train directory structure (alphabetical)
    train_candidates = [
        os.path.join(ROOT, "../dataset/preprocessed/Train"),
        os.path.join(ROOT, "../dataset/preprocessed/train"),
        os.path.join(ROOT, "../dataset/Train"),
        os.path.join(ROOT, "../dataset/train"),
    ]
    for td in train_candidates:
        if os.path.isdir(td):
            classes = sorted(
                [d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))]
            )
            if classes:
                return classes

    # 3) fallback to your current list
    return [
        'actinic keratosis',
        'basal cell carcinoma',
        'dermatofibroma',
        'melanoma',
        'nevus',
        'pigmented benign keratosis',
        'seborrheic keratosis',
        'squamous cell carcinoma',
        'vascular lesion'
    ]

class_names = load_class_names()

# Sanity check: class count vs model outputs
num_out = model.output_shape[-1] if isinstance(model.output_shape, tuple) else model.output_shape[0][-1]
if num_out != len(class_names):
    print(f"[WARN] Model outputs {num_out} classes but class_names has {len(class_names)}.")
    print("       Labels may be misaligned. Consider saving class_indices.json at training time.")

def predict_one(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x, verbose=0)[0]

    top3 = preds.argsort()[-3:][::-1]
    print(f"\nImage: {img_path}")
    for rank, i in enumerate(top3, 1):
        print(f"{rank}. {class_names[i]}: {preds[i]*100:.2f}%")
    print(f"Top-1 => {class_names[top3[0]]}\n")

def is_image(p):
    p_lower = p.lower()
    return p_lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path_or_folder>")
        sys.exit(0)

    target = sys.argv[1]
    if os.path.isdir(target):
        # Predict on all images in the folder
        files = [os.path.join(target, f) for f in os.listdir(target) if is_image(f)]
        if not files:
            print("No images found in the folder.")
        for f in sorted(files):
            predict_one(f)
    else:
        predict_one(target)

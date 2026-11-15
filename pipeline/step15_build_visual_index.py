import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# ============================================================
# CINEVISION AI - STEP 15
# BUILD REVERSE POSTER VISUAL EMBEDDINGS INDEX
# ============================================================

BASE_DIR = (
    Path(__file__).resolve().parent.parent
    if Path(__file__).resolve().parent.name == "pipeline"
    else Path(__file__).resolve().parent
)
EVAL_DIR = BASE_DIR / "evaluation"
SPLIT_DIR = BASE_DIR / "dataset" / "splits"
IMAGE_DIR = BASE_DIR / "dataset" / "master" / "images"
EFF_MODEL_PATH = BASE_DIR / "models" / "efficientnet" / "efficientnet_best.keras"

INDEX_OUT_PATH = EVAL_DIR / "visual_embeddings.npz"

TARGET_GENRES = [
    "Action",
    "Adventure",
    "Comedy",
    "Crime",
    "Drama",
    "Horror",
    "Romance",
    "Thriller"
]

print("=" * 70)
print("CINEVISION AI - STEP 15: BUILD VISUAL EMBEDDINGS INDEX")
print("=" * 70)

# Load test split (and validation split to enrich search database)
test_df = pd.read_csv(SPLIT_DIR / "test.csv")
val_df = pd.read_csv(SPLIT_DIR / "validation.csv")
combined_df = pd.concat([test_df, val_df], ignore_index=True).drop_duplicates(subset=["image_id"])

print(f"Indexing {len(combined_df)} reference movie posters into visual vector database...")

# Load EfficientNet model
print("\nLoading EfficientNet model for feature extraction...")
model = tf.keras.models.load_model(EFF_MODEL_PATH, compile=False)

# Build feature extractor to pooling layer
try:
    gap_layer = model.get_layer("global_average_pooling")
    feat_extractor = tf.keras.models.Model(inputs=model.inputs, outputs=gap_layer.output)
except Exception:
    feat_extractor = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-2].output)

image_ids = []
embeddings = []
genres_matrix = []

batch_imgs = []
batch_ids = []
batch_labels = []

BATCH_SIZE = 32

for idx, row in combined_df.iterrows():
    img_id = str(row["image_id"])
    img_path = IMAGE_DIR / f"{img_id}.jpg"
    if not img_path.exists():
        continue

    try:
        pil_img = Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        arr = np.array(pil_img, dtype=np.float32)
        batch_imgs.append(arr)
        batch_ids.append(img_id)
        batch_labels.append(row[TARGET_GENRES].values.astype(int))
    except Exception:
        continue

    if len(batch_imgs) >= BATCH_SIZE:
        t_arr = np.array(batch_imgs, dtype=np.float32)
        feats = feat_extractor(t_arr, training=False).numpy()
        # L2 normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms

        embeddings.append(feats)
        image_ids.extend(batch_ids)
        genres_matrix.extend(batch_labels)

        batch_imgs, batch_ids, batch_labels = [], [], []
        if len(image_ids) % 256 == 0:
            print(f"  Indexed {len(image_ids)} / {len(combined_df)} posters...")

if batch_imgs:
    t_arr = np.array(batch_imgs, dtype=np.float32)
    feats = feat_extractor(t_arr, training=False).numpy()
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats = feats / norms
    embeddings.append(feats)
    image_ids.extend(batch_ids)
    genres_matrix.extend(batch_labels)

all_embeddings = np.vstack(embeddings).astype(np.float32)
all_ids = np.array(image_ids)
all_genres = np.array(genres_matrix, dtype=int)

print(f"\nSuccessfully compiled visual vector index:")
print(f"  Embeddings Matrix: {all_embeddings.shape} (float32)")
print(f"  Indexed Movies:    {len(all_ids)}")

np.savez_compressed(
    INDEX_OUT_PATH,
    embeddings=all_embeddings,
    image_ids=all_ids,
    genres=all_genres
)

print(f"Saved visual vector index to:\n  {INDEX_OUT_PATH}")
print("STEP 15 COMPLETE.")


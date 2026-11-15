import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# ============================================================
# CINEVISION AI - REVERSE POSTER SEARCH & SIMILARITY UTILITY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "evaluation" / "visual_embeddings.npz"
MASTER_LABELS_PATH = BASE_DIR / "dataset" / "master" / "labels.csv"
MASTER_IMAGES_DIR = BASE_DIR / "dataset" / "master" / "images"

GENRES = [
    "Action",
    "Adventure",
    "Comedy",
    "Crime",
    "Drama",
    "Horror",
    "Romance",
    "Thriller"
]

def extract_poster_embedding(keras_model, pil_img):
    """
    Extracts the 1280-dimensional latent feature embedding from EfficientNet.
    """
    img_rgb = pil_img.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    img_arr = np.expand_dims(np.array(img_rgb, dtype=np.float32), axis=0)

    # Feature extractor submodel
    try:
        gap_layer = keras_model.get_layer("global_average_pooling")
        feat_extractor = tf.keras.models.Model(inputs=keras_model.inputs, outputs=gap_layer.output)
    except Exception:
        # Fallback to second-to-last layer
        feat_extractor = tf.keras.models.Model(inputs=keras_model.inputs, outputs=keras_model.layers[-2].output)

    embedding = feat_extractor(img_arr, training=False).numpy().flatten()
    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding

def find_similar_posters(query_embedding, top_k=5):
    """
    Finds the top-K nearest neighbor posters using cosine similarity over the cached index.
    """
    if not INDEX_PATH.exists():
        return []

    data = np.load(INDEX_PATH, allow_pickle=True)
    embeddings = data["embeddings"]  # shape (N, 1280)
    image_ids = data["image_ids"]
    genres_matrix = data["genres"]

    # Cosine similarities via dot product (since normalized)
    sims = np.dot(embeddings, query_embedding)

    # Top K indices
    top_indices = np.argsort(-sims)[:top_k]

    results = []
    for idx in top_indices:
        img_id = str(image_ids[idx])
        img_path = MASTER_IMAGES_DIR / f"{img_id}.jpg"
        
        # Determine genres with label=1
        labels = genres_matrix[idx]
        active_genres = [GENRES[i] for i, v in enumerate(labels) if v == 1]

        results.append({
            "image_id": img_id,
            "similarity": round(float(sims[idx]) * 100, 1),
            "similarity_raw": float(sims[idx]),
            "genres": active_genres if active_genres else ["Unknown"],
            "image_path": str(img_path) if img_path.exists() else None
        })

    return results


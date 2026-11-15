import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)


# ============================================================
# CINEVISION AI - STEP 6
# BASELINE CNN
# ============================================================

from pathlib import Path

BASE_DIR = str(
    Path(__file__).resolve().parent.parent
    if Path(__file__).resolve().parent.name == "pipeline"
    else Path(__file__).resolve().parent
)

MASTER_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "master"
)

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "splits"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "baseline_cnn"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 8

EPOCHS = 30

RANDOM_SEED = 42

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


# ============================================================
# REPRODUCIBILITY
# ============================================================

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# FILE PATHS
# ============================================================

IMAGE_DIR = os.path.join(
    MASTER_DIR,
    "images"
)

TRAIN_CSV = os.path.join(
    SPLIT_DIR,
    "train.csv"
)

VAL_CSV = os.path.join(
    SPLIT_DIR,
    "validation.csv"
)

TEST_CSV = os.path.join(
    SPLIT_DIR,
    "test.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

BEST_MODEL = os.path.join(
    MODEL_DIR,
    "baseline_cnn_best.keras"
)

FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "baseline_cnn_final.keras"
)

HISTORY_FILE = os.path.join(
    MODEL_DIR,
    "training_history.csv"
)

CONFIG_FILE = os.path.join(
    MODEL_DIR,
    "model_config.json"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CINEVISION AI - STEP 6 BASELINE CNN")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset splits...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"Training samples:   {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples:       {len(test_df)}")


# ============================================================
# IMAGE PATH FUNCTION
# ============================================================

def add_image_paths(df):

    df = df.copy()

    df["image_path"] = df["image_id"].apply(
        lambda image_id: os.path.join(
            IMAGE_DIR,
            f"{image_id}.jpg"
        )
    )

    return df


train_df = add_image_paths(train_df)
val_df = add_image_paths(val_df)
test_df = add_image_paths(test_df)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path, labels):

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = image / 255.0

    return image, labels


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.05
        ),

        layers.RandomZoom(
            height_factor=0.08,
            width_factor=0.08
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        layers.RandomContrast(
            0.10
        )

    ],
    name="poster_augmentation"
)


# ============================================================
# AUGMENTATION FUNCTION
# ============================================================

def augment_image(image, label):

    image = data_augmentation(
        image,
        training=True
    )

    image = tf.clip_by_value(
        image,
        0.0,
        1.0
    )

    return image, label


# ============================================================
# CREATE DATASET
# ============================================================

def create_dataset(
    dataframe,
    training=False
):

    image_paths = dataframe[
        "image_path"
    ].values

    labels = dataframe[
        TARGET_GENRES
    ].values.astype(
        "float32"
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels
        )
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:

        dataset = dataset.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# CREATE DATASETS
# ============================================================

print("\nCreating TensorFlow datasets...")

train_dataset = create_dataset(
    train_df,
    training=True
)

validation_dataset = create_dataset(
    val_df,
    training=False
)

test_dataset = create_dataset(
    test_df,
    training=False
)

print("Datasets created successfully.")


# ============================================================
# BUILD BASELINE CNN
# ============================================================

print("\nBuilding baseline CNN...")


model = models.Sequential(
    [

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        layers.Input(
            shape=(
                224,
                224,
                3
            )
        ),


        # ----------------------------------------------------
        # Block 1
        # ----------------------------------------------------

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),


        # ----------------------------------------------------
        # Block 2
        # ----------------------------------------------------

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),


        # ----------------------------------------------------
        # Block 3
        # ----------------------------------------------------

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),


        # ----------------------------------------------------
        # Block 4
        # ----------------------------------------------------

        layers.Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),


        # ----------------------------------------------------
        # Feature aggregation
        # ----------------------------------------------------

        layers.GlobalAveragePooling2D(),


        # ----------------------------------------------------
        # Fully connected layer
        # ----------------------------------------------------

        layers.Dense(
            256,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(
            0.40
        ),


        # ----------------------------------------------------
        # Multi-label output
        # ----------------------------------------------------

        layers.Dense(
            NUM_CLASSES,
            activation="sigmoid",
            name="genre_output"
        )

    ],
    name="CineVision_Baseline_CNN"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n")
model.summary()


# ============================================================
# COMPILE MODEL
# ============================================================

print("\nCompiling model...")

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss=tf.keras.losses.BinaryCrossentropy(),

    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="binary_accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc",
            multi_label=True,
            num_labels=NUM_CLASSES
        ),

        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR",
            multi_label=True,
            num_labels=NUM_CLASSES
        )
    ]
)

print("Model compiled successfully.")


# ============================================================
# SAVE MODEL CONFIGURATION
# ============================================================

model_config = {

    "model_name": "CineVision_Baseline_CNN",

    "task": "Multi-Label Movie Genre Classification",

    "input_size": [
        224,
        224,
        3
    ],

    "num_classes": NUM_CLASSES,

    "genres": TARGET_GENRES,

    "output_activation": "sigmoid",

    "loss": "binary_crossentropy",

    "optimizer": "Adam",

    "initial_learning_rate": 0.001,

    "batch_size": BATCH_SIZE,

    "maximum_epochs": EPOCHS,

    "augmentation": [
        "RandomFlip",
        "RandomRotation",
        "RandomZoom",
        "RandomTranslation",
        "RandomContrast"
    ]

}


with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_config,
        f,
        indent=4
    )


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    filepath=BEST_MODEL,

    monitor="val_pr_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_pr_auc",

    mode="max",

    patience=6,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_pr_auc",

    mode="max",

    factor=0.5,

    patience=3,

    min_lr=1e-6,

    verbose=1
)


csv_logger = CSVLogger(
    HISTORY_FILE
)


callbacks = [
    checkpoint,
    early_stopping,
    reduce_lr,
    csv_logger
]


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING BASELINE CNN TRAINING")
print("=" * 70)

print(f"\nMaximum epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print("Monitoring: validation PR-AUC")

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\nSaving final model...")

model.save(
    FINAL_MODEL
)

print(
    f"Final model saved to:\n{FINAL_MODEL}"
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nEpochs actually completed: "
    f"{len(history_df)}"
)


# ============================================================
# BEST VALIDATION RESULTS
# ============================================================

if "val_pr_auc" in history_df.columns:

    best_epoch = (
        history_df["val_pr_auc"]
        .idxmax()
    )

    best_val_pr_auc = (
        history_df
        .loc[best_epoch, "val_pr_auc"]
    )

    print(
        f"\nBest validation PR-AUC: "
        f"{best_val_pr_auc:.4f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch + 1}"
    )


# ============================================================
# FINAL VALIDATION RESULTS
# ============================================================

print("\nFinal validation metrics:")

for metric, value in history.history.items():

    if metric.startswith("val_"):

        print(
            f"{metric}: "
            f"{value[-1]:.4f}"
        )


# ============================================================
# FILES CREATED
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 FILES")
print("=" * 70)

print(
    f"\nBest model:\n{BEST_MODEL}"
)

print(
    f"\nFinal model:\n{FINAL_MODEL}"
)

print(
    f"\nTraining history:\n{HISTORY_FILE}"
)

print(
    f"\nModel configuration:\n{CONFIG_FILE}"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 BASELINE CNN TRAINING COMPLETE")
print("=" * 70)

print("\nSTATUS: PASS")
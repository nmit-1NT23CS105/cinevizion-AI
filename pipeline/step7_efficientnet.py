import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)


# ============================================================
# CINEVISION AI - STEP 7
# EFFICIENTNET TRANSFER LEARNING
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
    "efficientnet"
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

FEATURE_EPOCHS = 15

FINE_TUNE_EPOCHS = 20

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
# PATHS
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
    "efficientnet_best.keras"
)

FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "efficientnet_final.keras"
)

FEATURE_HISTORY = os.path.join(
    MODEL_DIR,
    "feature_training_history.csv"
)

FINETUNE_HISTORY = os.path.join(
    MODEL_DIR,
    "finetune_training_history.csv"
)

POS_WEIGHTS_FILE = os.path.join(
    MODEL_DIR,
    "positive_weights.json"
)

CONFIG_FILE = os.path.join(
    MODEL_DIR,
    "model_config.json"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CINEVISION AI - STEP 7 EFFICIENTNET")
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
# IMAGE PATHS
# ============================================================

def add_image_paths(dataframe):

    dataframe = dataframe.copy()

    dataframe["image_path"] = dataframe[
        "image_id"
    ].apply(
        lambda image_id: os.path.join(
            IMAGE_DIR,
            f"{image_id}.jpg"
        )
    )

    return dataframe


train_df = add_image_paths(train_df)
val_df = add_image_paths(val_df)
test_df = add_image_paths(test_df)


# ============================================================
# VERIFY IMAGES
# ============================================================

print("\nChecking image files...")

for name, dataframe in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    missing = [
        path
        for path in dataframe["image_path"]
        if not os.path.exists(path)
    ]

    print(
        f"{name}: {len(missing)} missing images"
    )

    if missing:
        raise FileNotFoundError(
            f"{name} contains missing images."
        )


print("All image files verified.")


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

    # EfficientNet preprocessing is included
    # in the Keras EfficientNet model.
    #
    # Therefore keep images in the [0,255] range here.

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
# AUGMENTATION
# ============================================================

def augment_image(image, label):

    image = data_augmentation(
        image,
        training=True
    )

    # Keep valid image range.
    image = tf.clip_by_value(
        image,
        0.0,
        255.0
    )

    return image, label


# ============================================================
# DATASET CREATION
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
# CALCULATE POSITIVE LABEL WEIGHTS
# ============================================================
#
# For each genre:
#
# positive_weight = negative_count / positive_count
#
# This gives more importance to underrepresented
# positive labels such as Horror.
#
# The weights are calculated ONLY from training data.
# ============================================================

print("\nCalculating training-set positive weights...")

train_labels_np = train_df[
    TARGET_GENRES
].values.astype(
    np.float32
)

positive_counts = train_labels_np.sum(
    axis=0
)

negative_counts = (
    len(train_labels_np)
    - positive_counts
)

positive_weights = (
    negative_counts /
    np.maximum(
        positive_counts,
        1
    )
)

print("\nPositive weights:")

for genre, pos, neg, weight in zip(
    TARGET_GENRES,
    positive_counts,
    negative_counts,
    positive_weights
):

    print(
        f"{genre:10s} "
        f"positive={int(pos):4d} "
        f"negative={int(neg):4d} "
        f"weight={weight:.4f}"
    )


# ============================================================
# SAVE POSITIVE WEIGHTS
# ============================================================

weights_dict = {
    genre: float(weight)
    for genre, weight in zip(
        TARGET_GENRES,
        positive_weights
    )
}

with open(
    POS_WEIGHTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        weights_dict,
        f,
        indent=4
    )


# ============================================================
# CUSTOM WEIGHTED BINARY CROSS ENTROPY
# ============================================================

class WeightedBinaryCrossEntropy(
    tf.keras.losses.Loss
):

    def __init__(
        self,
        positive_weights,
        name="weighted_binary_crossentropy"
    ):

        super().__init__(
            name=name
        )

        self.positive_weights = tf.constant(
            positive_weights,
            dtype=tf.float32
        )


    def call(
        self,
        y_true,
        y_pred
    ):

        # Prevent log(0)
        epsilon = tf.keras.backend.epsilon()

        y_pred = tf.clip_by_value(
            y_pred,
            epsilon,
            1.0 - epsilon
        )

        # Standard binary cross entropy
        bce = -(
            y_true * tf.math.log(y_pred)
            +
            (1.0 - y_true)
            * tf.math.log(1.0 - y_pred)
        )

        # Positive labels receive higher weight.
        weights = (
            y_true * self.positive_weights
            +
            (1.0 - y_true)
        )

        weighted_bce = (
            bce * weights
        )

        return tf.reduce_mean(
            weighted_bce
        )


# ============================================================
# LOSS
# ============================================================

weighted_loss = WeightedBinaryCrossEntropy(
    positive_weights
)


# ============================================================
# BUILD EFFICIENTNET
# ============================================================

print("\nBuilding EfficientNetB0...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        224,
        224,
        3
    )
)


# ============================================================
# PHASE A:
# FEATURE EXTRACTION
# ============================================================

base_model.trainable = False


inputs = layers.Input(
    shape=(
        224,
        224,
        3
    ),
    name="poster_input"
)


x = base_model(
    inputs,
    training=False
)


x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)


x = layers.Dense(
    256,
    activation="relu",
    name="classification_features"
)(x)


x = layers.BatchNormalization()(x)


x = layers.Dropout(
    0.40
)(x)


outputs = layers.Dense(
    NUM_CLASSES,
    activation="sigmoid",
    name="genre_output"
)(x)


model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="CineVision_EfficientNetB0"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel architecture:")

model.summary()


# ============================================================
# COMPILE PHASE A
# ============================================================

print("\nCompiling Phase A...")

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss=weighted_loss,

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


# ============================================================
# PHASE A CALLBACKS
# ============================================================

phase_a_best = os.path.join(
    MODEL_DIR,
    "efficientnet_phase_a_best.keras"
)

phase_a_checkpoint = ModelCheckpoint(

    filepath=phase_a_best,

    monitor="val_pr_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)


phase_a_early = EarlyStopping(

    monitor="val_pr_auc",

    mode="max",

    patience=4,

    restore_best_weights=True,

    verbose=1
)


phase_a_lr = ReduceLROnPlateau(

    monitor="val_pr_auc",

    mode="max",

    factor=0.5,

    patience=2,

    min_lr=1e-6,

    verbose=1
)


phase_a_logger = CSVLogger(
    FEATURE_HISTORY
)


# ============================================================
# PHASE A TRAINING
# ============================================================

print("\n" + "=" * 70)
print("PHASE A - FEATURE EXTRACTION")
print("=" * 70)

history_a = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=FEATURE_EPOCHS,

    callbacks=[
        phase_a_checkpoint,
        phase_a_early,
        phase_a_lr,
        phase_a_logger
    ],

    verbose=1
)


# ============================================================
# PHASE B:
# FINE-TUNING
# ============================================================

print("\n" + "=" * 70)
print("PHASE B - FINE-TUNING")
print("=" * 70)


# Unfreeze the base model
base_model.trainable = True


# ------------------------------------------------------------
# Freeze lower layers.
#
# EfficientNetB0 has many layers.
# We fine-tune only the upper portion.
# ------------------------------------------------------------

fine_tune_from = int(
    len(base_model.layers) * 0.70
)


for layer_index, layer in enumerate(
    base_model.layers
):

    if layer_index < fine_tune_from:

        layer.trainable = False

    else:

        layer.trainable = True


# ------------------------------------------------------------
# Keep BatchNormalization layers frozen.
#
# This is important for stable fine-tuning on a
# relatively small dataset.
# ------------------------------------------------------------

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


trainable_base_layers = sum(
    layer.trainable
    for layer in base_model.layers
)

print(
    f"\nEfficientNet layers: "
    f"{len(base_model.layers)}"
)

print(
    f"Trainable EfficientNet layers: "
    f"{trainable_base_layers}"
)


# ============================================================
# RECOMPILE WITH SMALL LEARNING RATE
# ============================================================

print("\nRecompiling for fine-tuning...")

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss=weighted_loss,

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


# ============================================================
# PHASE B CALLBACKS
# ============================================================

phase_b_checkpoint = ModelCheckpoint(

    filepath=BEST_MODEL,

    monitor="val_pr_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)


phase_b_early = EarlyStopping(

    monitor="val_pr_auc",

    mode="max",

    patience=6,

    restore_best_weights=True,

    verbose=1
)


phase_b_lr = ReduceLROnPlateau(

    monitor="val_pr_auc",

    mode="max",

    factor=0.5,

    patience=3,

    min_lr=1e-7,

    verbose=1
)


phase_b_logger = CSVLogger(
    FINETUNE_HISTORY
)


# ============================================================
# PHASE B TRAINING
# ============================================================

history_b = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=[
        phase_b_checkpoint,
        phase_b_early,
        phase_b_lr,
        phase_b_logger
    ],

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\nSaving final EfficientNet model...")

model.save(
    FINAL_MODEL
)

print(
    f"Final model saved to:\n{FINAL_MODEL}"
)


# ============================================================
# CONFIGURATION
# ============================================================

config = {

    "model_name":
        "CineVision_EfficientNetB0",

    "architecture":
        "EfficientNetB0",

    "pretrained_weights":
        "ImageNet",

    "input_size":
        [224, 224, 3],

    "num_classes":
        NUM_CLASSES,

    "genres":
        TARGET_GENRES,

    "output_activation":
        "sigmoid",

    "loss":
        "weighted_binary_crossentropy",

    "phase_a_epochs":
        FEATURE_EPOCHS,

    "phase_a_learning_rate":
        0.001,

    "phase_b_epochs":
        FINE_TUNE_EPOCHS,

    "phase_b_learning_rate":
        0.00001,

    "batch_size":
        BATCH_SIZE,

    "augmentation":
        [
            "RandomFlip",
            "RandomRotation",
            "RandomZoom",
            "RandomTranslation",
            "RandomContrast"
        ],

    "fine_tuning":
        "Upper 30 percent of EfficientNet layers",

    "batch_normalization_layers":
        "Frozen"

}


with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# TRAINING HISTORY SUMMARY
# ============================================================

history_a_df = pd.DataFrame(
    history_a.history
)

history_b_df = pd.DataFrame(
    history_b.history
)


print("\n" + "=" * 70)
print("EFFICIENTNET TRAINING COMPLETED")
print("=" * 70)


# ============================================================
# PHASE A RESULT
# ============================================================

if "val_pr_auc" in history_a_df.columns:

    best_a_index = (
        history_a_df[
            "val_pr_auc"
        ].idxmax()
    )

    best_a_pr_auc = (
        history_a_df.loc[
            best_a_index,
            "val_pr_auc"
        ]
    )

    print(
        f"\nPhase A best validation PR-AUC: "
        f"{best_a_pr_auc:.4f}"
    )

    print(
        f"Phase A best epoch: "
        f"{best_a_index + 1}"
    )


# ============================================================
# PHASE B RESULT
# ============================================================

if "val_pr_auc" in history_b_df.columns:

    best_b_index = (
        history_b_df[
            "val_pr_auc"
        ].idxmax()
    )

    best_b_pr_auc = (
        history_b_df.loc[
            best_b_index,
            "val_pr_auc"
        ]
    )

    print(
        f"\nPhase B best validation PR-AUC: "
        f"{best_b_pr_auc:.4f}"
    )

    print(
        f"Phase B best epoch: "
        f"{best_b_index + 1}"
    )


# ============================================================
# FINAL VALIDATION METRICS
# ============================================================

print("\nFinal validation metrics:")

for metric, values in history_b.history.items():

    if metric.startswith(
        "val_"
    ):

        print(
            f"{metric}: "
            f"{values[-1]:.4f}"
        )


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    f"\nBest model:\n{BEST_MODEL}"
)

print(
    f"\nFinal model:\n{FINAL_MODEL}"
)

print(
    f"\nPhase A history:\n{FEATURE_HISTORY}"
)

print(
    f"\nPhase B history:\n{FINETUNE_HISTORY}"
)

print(
    f"\nPositive weights:\n{POS_WEIGHTS_FILE}"
)

print(
    f"\nConfiguration:\n{CONFIG_FILE}"
)


# ============================================================
# STATUS
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 EFFICIENTNET TRAINING COMPLETE")
print("=" * 70)

print("\nSTATUS: PASS")
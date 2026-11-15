import os
import tensorflow as tf
import pandas as pd


# ============================================================
# CINEVISION AI - STEP 5
# IMAGE PREPROCESSING & AUGMENTATION PIPELINE
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


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

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
# START
# ============================================================

print("=" * 70)
print("CINEVISION AI - STEP 5 PREPROCESSING PIPELINE")
print("=" * 70)


# ============================================================
# LOAD CSV FILES
# ============================================================

print("\nLoading dataset splits...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print("\nDataset sizes:")
print(f"Training:   {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_id"
] + TARGET_GENRES

for name, dataframe in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: {missing_columns}"
        )

print("\nAll required columns found.")


# ============================================================
# VERIFY LABEL VALUES
# ============================================================

for name, dataframe in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    labels = dataframe[
        TARGET_GENRES
    ].values.astype(int)

    if not all(
        value in [0, 1]
        for value in labels.flatten()
    ):
        raise ValueError(
            f"{name} contains labels other than 0 or 1."
        )

    if (labels.sum(axis=1) == 0).any():
        raise ValueError(
            f"{name} contains images with zero target genres."
        )

print("Label validation passed.")


# ============================================================
# CREATE IMAGE PATHS
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
# VERIFY IMAGE FILES
# ============================================================

print("\nChecking image paths...")

for name, dataframe in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    missing_images = [
        path
        for path in dataframe["image_path"]
        if not os.path.exists(path)
    ]

    print(
        f"{name}: {len(missing_images)} missing images"
    )

    if missing_images:

        print("\nFirst missing images:")

        for path in missing_images[:10]:
            print(path)

        raise FileNotFoundError(
            f"{name} contains missing image files."
        )

print("All image paths verified.")


# ============================================================
# IMAGE LOADING FUNCTION
# ============================================================

def load_image(image_path, labels):

    # Read image file
    image = tf.io.read_file(
        image_path
    )

    # Decode JPG
    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    # Resize to 224 x 224
    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    # Convert to float32
    image = tf.cast(
        image,
        tf.float32
    )

    # Normalize pixel values
    # 0-255 -> 0-1
    image = image / 255.0

    return image, labels


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [

        # Random horizontal flip
        tf.keras.layers.RandomFlip(
            mode="horizontal"
        ),

        # Small rotation
        tf.keras.layers.RandomRotation(
            factor=0.05
        ),

        # Small zoom
        tf.keras.layers.RandomZoom(
            height_factor=0.08,
            width_factor=0.08
        ),

        # Small translation
        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        # Mild contrast adjustment
        tf.keras.layers.RandomContrast(
            factor=0.10
        )

    ],
    name="poster_augmentation"
)


# ============================================================
# AUGMENTATION FUNCTION
# ============================================================

def augment_image(image, label):

    # Apply augmentation
    image = data_augmentation(
        image,
        training=True
    )

    # Some augmentation operations can produce
    # values slightly outside [0,1].
    #
    # Clip them back into the valid pixel range.
    image = tf.clip_by_value(
        image,
        0.0,
        1.0
    )

    return image, label


# ============================================================
# CREATE TENSORFLOW DATASET
# ============================================================

def create_dataset(
    dataframe,
    training=False
):

    # --------------------------------------------------------
    # Get image paths
    # --------------------------------------------------------

    image_paths = dataframe[
        "image_path"
    ].values

    # --------------------------------------------------------
    # Get multi-label targets
    # --------------------------------------------------------

    labels = dataframe[
        TARGET_GENRES
    ].values.astype(
        "float32"
    )

    # --------------------------------------------------------
    # Create TensorFlow dataset
    # --------------------------------------------------------

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels
        )
    )

    # --------------------------------------------------------
    # Shuffle ONLY training data
    # --------------------------------------------------------

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=42,
            reshuffle_each_iteration=True
        )

    # --------------------------------------------------------
    # Load and preprocess images
    # --------------------------------------------------------

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # Apply augmentation ONLY to training data
    # --------------------------------------------------------

    if training:

        dataset = dataset.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    dataset = dataset.batch(
        BATCH_SIZE
    )

    # --------------------------------------------------------
    # Prefetch
    # --------------------------------------------------------

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# CREATE TRAINING DATASET
# ============================================================

print("\nCreating TensorFlow datasets...")

train_dataset = create_dataset(
    train_df,
    training=True
)


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

validation_dataset = create_dataset(
    val_df,
    training=False
)


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_dataset = create_dataset(
    test_df,
    training=False
)


# ============================================================
# INSPECT TRAINING BATCH
# ============================================================

print("\nInspecting training batch...")

train_images, train_labels = next(
    iter(train_dataset)
)

print(
    f"Image batch shape: {train_images.shape}"
)

print(
    f"Label batch shape: {train_labels.shape}"
)

print(
    f"Image dtype: {train_images.dtype}"
)

print(
    f"Label dtype: {train_labels.dtype}"
)

train_min = tf.reduce_min(
    train_images
).numpy()

train_max = tf.reduce_max(
    train_images
).numpy()

print(
    f"Pixel minimum: {train_min:.4f}"
)

print(
    f"Pixel maximum: {train_max:.4f}"
)


# ============================================================
# INSPECT VALIDATION BATCH
# ============================================================

print("\nInspecting validation batch...")

val_images, val_labels = next(
    iter(validation_dataset)
)

print(
    f"Validation image shape: {val_images.shape}"
)

print(
    f"Validation label shape: {val_labels.shape}"
)

val_min = tf.reduce_min(
    val_images
).numpy()

val_max = tf.reduce_max(
    val_images
).numpy()

print(
    f"Validation pixel minimum: {val_min:.4f}"
)

print(
    f"Validation pixel maximum: {val_max:.4f}"
)


# ============================================================
# INSPECT TEST BATCH
# ============================================================

print("\nInspecting test batch...")

test_images, test_labels = next(
    iter(test_dataset)
)

print(
    f"Test image shape: {test_images.shape}"
)

print(
    f"Test label shape: {test_labels.shape}"
)

test_min = tf.reduce_min(
    test_images
).numpy()

test_max = tf.reduce_max(
    test_images
).numpy()

print(
    f"Test pixel minimum: {test_min:.4f}"
)

print(
    f"Test pixel maximum: {test_max:.4f}"
)


# ============================================================
# VERIFY LABEL DIMENSIONS
# ============================================================

print("\nVerifying label dimensions...")

if train_labels.shape[1] != 8:

    raise ValueError(
        "Training labels do not contain exactly 8 genres."
    )

if val_labels.shape[1] != 8:

    raise ValueError(
        "Validation labels do not contain exactly 8 genres."
    )

if test_labels.shape[1] != 8:

    raise ValueError(
        "Test labels do not contain exactly 8 genres."
    )

print("Label dimensions: PASS")


# ============================================================
# VERIFY IMAGE DIMENSIONS
# ============================================================

print("\nVerifying image dimensions...")

expected_shape = (
    224,
    224,
    3
)

if train_images.shape[1:] != expected_shape:

    raise ValueError(
        "Training images do not have shape 224x224x3."
    )

if val_images.shape[1:] != expected_shape:

    raise ValueError(
        "Validation images do not have shape 224x224x3."
    )

if test_images.shape[1:] != expected_shape:

    raise ValueError(
        "Test images do not have shape 224x224x3."
    )

print("Image dimensions: PASS")


# ============================================================
# VERIFY PIXEL RANGE
# ============================================================

print("\nVerifying pixel ranges...")

if train_min < 0.0 or train_max > 1.0:

    raise ValueError(
        "Training images contain pixel values outside [0,1]."
    )

if val_min < 0.0 or val_max > 1.0:

    raise ValueError(
        "Validation images contain pixel values outside [0,1]."
    )

if test_min < 0.0 or test_max > 1.0:

    raise ValueError(
        "Test images contain pixel values outside [0,1]."
    )

print("Pixel range: PASS")


# ============================================================
# VERIFY DATASET BATCHES
# ============================================================

print("\nVerifying dataset iteration...")

train_batch_count = 0

for images, labels in train_dataset.take(2):

    train_batch_count += 1

    if images.shape[-1] != 3:
        raise ValueError(
            "Training batch does not contain RGB images."
        )

    if labels.shape[-1] != 8:
        raise ValueError(
            "Training batch does not contain 8 labels."
        )


val_batch_count = 0

for images, labels in validation_dataset.take(2):

    val_batch_count += 1

    if images.shape[-1] != 3:
        raise ValueError(
            "Validation batch does not contain RGB images."
        )

    if labels.shape[-1] != 8:
        raise ValueError(
            "Validation batch does not contain 8 labels."
        )


test_batch_count = 0

for images, labels in test_dataset.take(2):

    test_batch_count += 1

    if images.shape[-1] != 3:
        raise ValueError(
            "Test batch does not contain RGB images."
        )

    if labels.shape[-1] != 8:
        raise ValueError(
            "Test batch does not contain 8 labels."
        )

print("Dataset iteration: PASS")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 PREPROCESSING VALIDATION PASSED")
print("=" * 70)

print("\nImage configuration:")
print("--------------------")
print("Resolution: 224 x 224")
print("Channels: 3 (RGB)")
print("Data type: float32")
print("Pixel range: [0, 1]")


print("\nTraining configuration:")
print("-----------------------")
print("Shuffle: YES")
print("Augmentation: YES")
print("Batch size:", BATCH_SIZE)
print("Prefetch: YES")


print("\nValidation configuration:")
print("-------------------------")
print("Shuffle: NO")
print("Augmentation: NO")
print("Batch size:", BATCH_SIZE)
print("Prefetch: YES")


print("\nTest configuration:")
print("-------------------")
print("Shuffle: NO")
print("Augmentation: NO")
print("Batch size:", BATCH_SIZE)
print("Prefetch: YES")


print("\nAugmentation operations:")
print("------------------------")
print("Random horizontal flip")
print("Random rotation")
print("Random zoom")
print("Random translation")
print("Random contrast")
print("Pixel clipping to [0,1]")


print("\nMulti-label configuration:")
print("--------------------------")
print("Number of genres: 8")
print("Encoding: 0 / 1")
print("Output activation later: sigmoid")
print("Loss later: binary cross-entropy")


print("\nTensorFlow pipeline:")
print("-------------------")
print("tf.data.Dataset")
print("Parallel image loading")
print("Batching")
print("Prefetching")


print("\nSTATUS: PASS")
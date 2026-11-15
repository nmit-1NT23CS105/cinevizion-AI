import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    hamming_loss
)


# ============================================================
# CINEVISION AI - STEP 8
# MODEL EVALUATION & THRESHOLD OPTIMIZATION
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

EVAL_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)

os.makedirs(
    EVAL_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

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

BEST_MODEL = os.path.join(
    MODEL_DIR,
    "efficientnet_best.keras"
)


# ============================================================
# OUTPUT FILES
# ============================================================

THRESHOLD_FILE = os.path.join(
    EVAL_DIR,
    "optimal_thresholds.json"
)

VALIDATION_RESULTS = os.path.join(
    EVAL_DIR,
    "validation_metrics.csv"
)

TEST_RESULTS = os.path.join(
    EVAL_DIR,
    "test_metrics.csv"
)

VALIDATION_PREDICTIONS = os.path.join(
    EVAL_DIR,
    "validation_predictions.csv"
)

TEST_PREDICTIONS = os.path.join(
    EVAL_DIR,
    "test_predictions.csv"
)

SUMMARY_FILE = os.path.join(
    EVAL_DIR,
    "evaluation_summary.txt"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CINEVISION AI - STEP 8 MODEL EVALUATION")
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

    return image, labels


# ============================================================
# CREATE EVALUATION DATASET
# ============================================================

def create_eval_dataset(dataframe):

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

    dataset = dataset.map(
        load_image,
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
# CREATE VALIDATION & TEST DATASETS
# ============================================================

print("\nCreating evaluation datasets...")

validation_dataset = create_eval_dataset(
    val_df
)

test_dataset = create_eval_dataset(
    test_df
)

print("Evaluation datasets created.")


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best EfficientNet model...")

model = tf.keras.models.load_model(
    BEST_MODEL,
    compile=False
)

print(
    f"Loaded model:\n{BEST_MODEL}"
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

validation_probabilities = model.predict(
    validation_dataset,
    verbose=1
)

print(
    "\nValidation prediction shape:",
    validation_probabilities.shape
)


print("\nGenerating test predictions...")

test_probabilities = model.predict(
    test_dataset,
    verbose=1
)

print(
    "Test prediction shape:",
    test_probabilities.shape
)


# ============================================================
# TRUE LABELS
# ============================================================

y_val = val_df[
    TARGET_GENRES
].values.astype(
    int
)

y_test = test_df[
    TARGET_GENRES
].values.astype(
    int
)


# ============================================================
# BASIC SHAPE CHECK
# ============================================================

if validation_probabilities.shape != y_val.shape:

    raise ValueError(
        "Validation prediction shape does not match labels."
    )

if test_probabilities.shape != y_test.shape:

    raise ValueError(
        "Test prediction shape does not match labels."
    )


# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

print("\nSaving probability predictions...")

validation_prediction_df = pd.DataFrame(
    validation_probabilities,
    columns=[
        f"{genre}_probability"
        for genre in TARGET_GENRES
    ]
)

validation_prediction_df.insert(
    0,
    "image_id",
    val_df["image_id"].values
)

validation_prediction_df.to_csv(
    VALIDATION_PREDICTIONS,
    index=False
)


test_prediction_df = pd.DataFrame(
    test_probabilities,
    columns=[
        f"{genre}_probability"
        for genre in TARGET_GENRES
    ]
)

test_prediction_df.insert(
    0,
    "image_id",
    test_df["image_id"].values
)

test_prediction_df.to_csv(
    TEST_PREDICTIONS,
    index=False
)


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================
#
# We optimize thresholds ONLY using validation data.
#
# Test data is NOT used here.
# ============================================================

print("\n" + "=" * 70)
print("OPTIMIZING PER-GENRE THRESHOLDS")
print("=" * 70)

thresholds = {}

threshold_candidates = np.arange(
    0.10,
    0.91,
    0.01
)


for genre_index, genre in enumerate(
    TARGET_GENRES
):

    true_labels = y_val[
        :,
        genre_index
    ]

    probabilities = validation_probabilities[
        :,
        genre_index
    ]

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in threshold_candidates:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            true_labels,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = float(
                threshold
            )

    thresholds[genre] = best_threshold

    print(
        f"{genre:10s} "
        f"threshold={best_threshold:.2f} "
        f"validation_F1={best_f1:.4f}"
    )


# ============================================================
# SAVE THRESHOLDS
# ============================================================

with open(
    THRESHOLD_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        thresholds,
        f,
        indent=4
    )


print(
    f"\nThresholds saved to:\n{THRESHOLD_FILE}"
)


# ============================================================
# APPLY THRESHOLDS
# ============================================================

threshold_array = np.array(
    [
        thresholds[genre]
        for genre in TARGET_GENRES
    ]
)


validation_predictions = (
    validation_probabilities
    >= threshold_array
).astype(int)


test_predictions = (
    test_probabilities
    >= threshold_array
).astype(int)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    predictions
):

    results = []

    for index, genre in enumerate(
        TARGET_GENRES
    ):

        true_labels = y_true[
            :,
            index
        ]

        probs = probabilities[
            :,
            index
        ]

        preds = predictions[
            :,
            index
        ]

        precision = precision_score(
            true_labels,
            preds,
            zero_division=0
        )

        recall = recall_score(
            true_labels,
            preds,
            zero_division=0
        )

        f1 = f1_score(
            true_labels,
            preds,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            true_labels,
            probs
        )

        pr_auc = average_precision_score(
            true_labels,
            probs
        )

        results.append(
            {
                "Genre": genre,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc
            }
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# VALIDATION METRICS
# ============================================================

print("\nCalculating validation metrics...")

validation_metrics = calculate_metrics(
    y_val,
    validation_probabilities,
    validation_predictions
)


# ============================================================
# TEST METRICS
# ============================================================

print("Calculating test metrics...")

test_metrics = calculate_metrics(
    y_test,
    test_probabilities,
    test_predictions
)


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_overall_metrics(
    y_true,
    probabilities,
    predictions
):

    macro_precision = precision_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    micro_precision = precision_score(
        y_true,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_recall = recall_score(
        y_true,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_true,
        predictions,
        average="micro",
        zero_division=0
    )

    hamming = hamming_loss(
        y_true,
        predictions
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
        average="macro"
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
        average="macro"
    )

    # Exact Match Ratio
    exact_match = np.mean(
        np.all(
            y_true == predictions,
            axis=1
        )
    )

    return {
        "Macro Precision":
            macro_precision,

        "Macro Recall":
            macro_recall,

        "Macro F1":
            macro_f1,

        "Micro Precision":
            micro_precision,

        "Micro Recall":
            micro_recall,

        "Micro F1":
            micro_f1,

        "Macro ROC-AUC":
            roc_auc,

        "Macro PR-AUC":
            pr_auc,

        "Hamming Loss":
            hamming,

        "Exact Match Ratio":
            exact_match
    }


validation_overall = calculate_overall_metrics(
    y_val,
    validation_probabilities,
    validation_predictions
)

test_overall = calculate_overall_metrics(
    y_test,
    test_probabilities,
    test_predictions
)


# ============================================================
# SAVE PER-GENRE RESULTS
# ============================================================

validation_metrics.to_csv(
    VALIDATION_RESULTS,
    index=False
)

test_metrics.to_csv(
    TEST_RESULTS,
    index=False
)


# ============================================================
# PRINT VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(
    validation_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nOverall validation metrics:")

for metric, value in validation_overall.items():

    print(
        f"{metric:25s}: {value:.4f}"
    )


# ============================================================
# PRINT TEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    test_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nOverall test metrics:")

for metric, value in test_overall.items():

    print(
        f"{metric:25s}: {value:.4f}"
    )


# ============================================================
# CREATE SUMMARY REPORT
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CINEVISION AI - MODEL EVALUATION SUMMARY\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Model: EfficientNetB0\n"
    )

    f.write(
        "Task: Multi-Label Movie Genre Classification\n\n"
    )

    f.write(
        "Threshold optimization:\n"
    )

    f.write(
        "Thresholds optimized ONLY on validation set.\n"
    )

    f.write(
        "Test set was not used during threshold optimization.\n\n"
    )

    f.write(
        "OPTIMAL THRESHOLDS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for genre, threshold in thresholds.items():

        f.write(
            f"{genre}: {threshold:.2f}\n"
        )

    f.write("\n")

    f.write(
        "VALIDATION OVERALL METRICS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for metric, value in validation_overall.items():

        f.write(
            f"{metric}: {value:.4f}\n"
        )

    f.write("\n")

    f.write(
        "TEST OVERALL METRICS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for metric, value in test_overall.items():

        f.write(
            f"{metric}: {value:.4f}\n"
        )

    f.write("\n")

    f.write(
        "TEST PER-GENRE METRICS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        test_metrics.to_string(
            index=False
        )
    )

    f.write("\n")


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 EVALUATION COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(
    f"\nOptimal thresholds:\n{THRESHOLD_FILE}"
)

print(
    f"\nValidation metrics:\n{VALIDATION_RESULTS}"
)

print(
    f"\nTest metrics:\n{TEST_RESULTS}"
)

print(
    f"\nValidation predictions:\n"
    f"{VALIDATION_PREDICTIONS}"
)

print(
    f"\nTest predictions:\n"
    f"{TEST_PREDICTIONS}"
)

print(
    f"\nEvaluation summary:\n{SUMMARY_FILE}"
)

print("\nSTATUS: PASS")
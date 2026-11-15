import os
import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# ============================================================
# CINEVISION AI - STEP 4
# Multi-Label Stratified Dataset Splitting
# ============================================================

from pathlib import Path

BASE_DIR = str(
    Path(__file__).resolve().parent.parent
    if Path(__file__).resolve().parent.name == "pipeline"
    else Path(__file__).resolve().parent
)

MASTER_DIR = os.path.join(BASE_DIR, "dataset", "master")
LABELS_FILE = os.path.join(MASTER_DIR, "labels.csv")

SPLIT_DIR = os.path.join(BASE_DIR, "dataset", "splits")

os.makedirs(SPLIT_DIR, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

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

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_STATE = 42


# ============================================================
# Load master dataset
# ============================================================

print("=" * 70)
print("CINEVISION AI - STEP 4 DATASET SPLITTING")
print("=" * 70)

print("\nLoading master dataset...")

df = pd.read_csv(LABELS_FILE)

print(f"Total records: {len(df)}")


# ============================================================
# Verify columns
# ============================================================

required_columns = ["image_id"] + TARGET_GENRES

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns found.")


# ============================================================
# Verify labels
# ============================================================

label_matrix = df[TARGET_GENRES].values.astype(int)

if not np.all(np.isin(label_matrix, [0, 1])):
    raise ValueError(
        "Target genre columns must contain only 0 and 1."
    )

if np.any(label_matrix.sum(axis=1) == 0):
    raise ValueError(
        "Dataset contains samples with zero target genres."
    )

print("Label validation passed.")


# ============================================================
# Verify image existence
# ============================================================

print("\nChecking image files...")

missing_images = []

for image_id in df["image_id"]:
    image_path = os.path.join(
        MASTER_DIR,
        "images",
        f"{image_id}.jpg"
    )

    if not os.path.exists(image_path):
        missing_images.append(image_id)

if missing_images:
    print(f"ERROR: {len(missing_images)} images are missing.")

    print("First missing images:")
    for image_id in missing_images[:10]:
        print(image_id)

    raise FileNotFoundError(
        "Some master dataset images are missing."
    )

print("All images verified.")


# ============================================================
# STEP 1
# Split test set = 10%
# ============================================================

print("\nCreating stratified test split...")

msss_test = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=TEST_RATIO,
    random_state=RANDOM_STATE
)

train_val_indices, test_indices = next(
    msss_test.split(
        df["image_id"],
        label_matrix
    )
)

train_val_df = df.iloc[train_val_indices].reset_index(drop=True)
test_df = df.iloc[test_indices].reset_index(drop=True)


# ============================================================
# STEP 2
# Split validation from remaining 90%
#
# Desired:
# Train = 80%
# Validation = 10%
# Test = 10%
#
# Validation fraction inside remaining 90%:
# 10 / 90 = 0.111111
# ============================================================

print("Creating stratified validation split...")

validation_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

train_val_labels = train_val_df[TARGET_GENRES].values.astype(int)

msss_val = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=validation_fraction,
    random_state=RANDOM_STATE
)

train_indices, validation_indices = next(
    msss_val.split(
        train_val_df["image_id"],
        train_val_labels
    )
)

train_df = train_val_df.iloc[train_indices].reset_index(drop=True)
validation_df = train_val_df.iloc[validation_indices].reset_index(drop=True)


# ============================================================
# Sort datasets by image_id
# ============================================================

train_df = train_df.sort_values("image_id").reset_index(drop=True)
validation_df = validation_df.sort_values("image_id").reset_index(drop=True)
test_df = test_df.sort_values("image_id").reset_index(drop=True)


# ============================================================
# Verify no leakage
# ============================================================

train_ids = set(train_df["image_id"])
validation_ids = set(validation_df["image_id"])
test_ids = set(test_df["image_id"])

train_val_overlap = train_ids.intersection(validation_ids)
train_test_overlap = train_ids.intersection(test_ids)
val_test_overlap = validation_ids.intersection(test_ids)

if train_val_overlap:
    raise ValueError("Data leakage detected: train/validation overlap.")

if train_test_overlap:
    raise ValueError("Data leakage detected: train/test overlap.")

if val_test_overlap:
    raise ValueError("Data leakage detected: validation/test overlap.")

print("\nNo image leakage detected.")


# ============================================================
# Verify complete dataset coverage
# ============================================================

combined_ids = train_ids | validation_ids | test_ids
master_ids = set(df["image_id"])

if combined_ids != master_ids:
    raise ValueError(
        "Split datasets do not exactly match master dataset."
    )

print("All master records are included exactly once.")


# ============================================================
# Save CSV files
# ============================================================

train_file = os.path.join(SPLIT_DIR, "train.csv")
validation_file = os.path.join(SPLIT_DIR, "validation.csv")
test_file = os.path.join(SPLIT_DIR, "test.csv")

train_df.to_csv(train_file, index=False)
validation_df.to_csv(validation_file, index=False)
test_df.to_csv(test_file, index=False)


# ============================================================
# Distribution report
# ============================================================

def distribution_table(dataframe):

    counts = dataframe[TARGET_GENRES].sum()

    percentages = (
        counts / len(dataframe) * 100
    )

    result = pd.DataFrame({
        "Genre": TARGET_GENRES,
        "Count": [
            int(counts[g])
            for g in TARGET_GENRES
        ],
        "Percentage": [
            round(percentages[g], 2)
            for g in TARGET_GENRES
        ]
    })

    return result


train_distribution = distribution_table(train_df)
validation_distribution = distribution_table(validation_df)
test_distribution = distribution_table(test_df)


# ============================================================
# Create report
# ============================================================

report_file = os.path.join(
    SPLIT_DIR,
    "split_report.txt"
)

with open(report_file, "w", encoding="utf-8") as f:

    f.write("CINEVISION AI - DATASET SPLIT REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write("Dataset: CineVision AI\n")
    f.write("Task: Multi-Label Movie Genre Classification\n\n")

    f.write("Target Genres:\n")

    for genre in TARGET_GENRES:
        f.write(f"- {genre}\n")

    f.write("\n")

    f.write("TOTAL RECORDS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Master:       {len(df)}\n")
    f.write(f"Training:     {len(train_df)}\n")
    f.write(f"Validation:   {len(validation_df)}\n")
    f.write(f"Test:         {len(test_df)}\n\n")

    f.write("SPLIT RATIOS\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Training:   {len(train_df) / len(df) * 100:.2f}%\n"
    )

    f.write(
        f"Validation: {len(validation_df) / len(df) * 100:.2f}%\n"
    )

    f.write(
        f"Test:       {len(test_df) / len(df) * 100:.2f}%\n\n"
    )

    f.write("TRAINING DISTRIBUTION\n")
    f.write("-" * 70 + "\n")
    f.write(
        train_distribution.to_string(index=False)
    )
    f.write("\n\n")

    f.write("VALIDATION DISTRIBUTION\n")
    f.write("-" * 70 + "\n")
    f.write(
        validation_distribution.to_string(index=False)
    )
    f.write("\n\n")

    f.write("TEST DISTRIBUTION\n")
    f.write("-" * 70 + "\n")
    f.write(
        test_distribution.to_string(index=False)
    )
    f.write("\n\n")

    f.write("DATA LEAKAGE CHECK\n")
    f.write("-" * 70 + "\n")
    f.write("Train ∩ Validation: 0\n")
    f.write("Train ∩ Test:       0\n")
    f.write("Validation ∩ Test:  0\n")
    f.write("Status: PASS\n")


# ============================================================
# Print final results
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nMaster records:      {len(df)}")
print(f"Training records:    {len(train_df)}")
print(f"Validation records:  {len(validation_df)}")
print(f"Test records:        {len(test_df)}")

print("\nActual split:")
print(
    f"Train:      {len(train_df) / len(df) * 100:.2f}%"
)
print(
    f"Validation: {len(validation_df) / len(df) * 100:.2f}%"
)
print(
    f"Test:       {len(test_df) / len(df) * 100:.2f}%"
)

print("\n" + "-" * 70)
print("TRAIN DISTRIBUTION")
print("-" * 70)
print(train_distribution.to_string(index=False))

print("\n" + "-" * 70)
print("VALIDATION DISTRIBUTION")
print("-" * 70)
print(validation_distribution.to_string(index=False))

print("\n" + "-" * 70)
print("TEST DISTRIBUTION")
print("-" * 70)
print(test_distribution.to_string(index=False))

print("\n" + "-" * 70)
print("FILES CREATED")
print("-" * 70)

print(train_file)
print(validation_file)
print(test_file)
print(report_file)

print("\nNo images were moved or modified.")
print("No raw datasets were modified.")

print("\nSTATUS: PASS")
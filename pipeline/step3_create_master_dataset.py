import os
import shutil
import pandas as pd
from PIL import Image

# ============================================================
# CINEVISION AI
# STEP 3 — CREATE CLEAN MASTER DATASET
#
# IMPORTANT:
# This script DOES NOT modify the original dataset.
# It only READS from Multi_Label_dataset and creates a
# separate CineVision_AI dataset.
# ============================================================


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
    if os.path.basename(os.path.dirname(__file__)) == "pipeline"
    else os.path.join(os.path.dirname(__file__), "..")
)

SOURCE_DIR = os.path.join(
    PROJECT_ROOT,
    "Multi_Label_dataset"
)

SOURCE_CSV = os.path.join(
    SOURCE_DIR,
    "train.csv"
)

SOURCE_IMAGES = os.path.join(
    SOURCE_DIR,
    "Images"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "CineVision_AI",
    "dataset",
    "master"
)

OUTPUT_IMAGES = os.path.join(
    OUTPUT_DIR,
    "images"
)

OUTPUT_LABELS = os.path.join(
    OUTPUT_DIR,
    "labels.csv"
)

OUTPUT_REPORT = os.path.join(
    OUTPUT_DIR,
    "dataset_report.txt"
)


# ============================================================
# 2. FINAL 8 GENRES
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


# ============================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_IMAGES,
    exist_ok=True
)


# ============================================================
# 4. LOAD ORIGINAL CSV
# ============================================================

print("=" * 70)
print("CINEVISION AI — STEP 3")
print("CREATING CLEAN MASTER DATASET")
print("=" * 70)

print("\nLoading source CSV...")

df = pd.read_csv(SOURCE_CSV)

print("Original records:", len(df))


# ============================================================
# 5. VERIFY REQUIRED COLUMNS
# ============================================================

print("\nChecking target genre columns...")

missing_columns = [
    genre
    for genre in TARGET_GENRES
    if genre not in df.columns
]

if missing_columns:

    print("\nERROR!")
    print(
        "Missing target columns:",
        missing_columns
    )

    raise SystemExit(
        "Dataset does not contain all 8 target genres."
    )

print("PASS: All 8 target genres found.")


# ============================================================
# 6. REMOVE RECORDS WITH NO TARGET GENRE
# ============================================================

print("\nRemoving records with zero target genres...")

df["_target_count"] = (
    df[TARGET_GENRES]
    .sum(axis=1)
)

before_no_target = len(df)

df = df[
    df["_target_count"] > 0
].copy()

removed_no_target = (
    before_no_target - len(df)
)

print(
    "Removed:",
    removed_no_target
)

print(
    "Remaining:",
    len(df)
)


# ============================================================
# 7. REMOVE DUPLICATE MOVIE IDS
# ============================================================

print("\nChecking duplicate movie IDs...")

before_duplicates = len(df)

duplicate_count = (
    df["Id"].duplicated()
    .sum()
)

print(
    "Duplicate records found:",
    duplicate_count
)

# Keep the first occurrence.
# Earlier audit confirmed duplicate IDs have
# identical labels.

df = df.drop_duplicates(
    subset="Id",
    keep="first"
).copy()

removed_duplicates = (
    before_duplicates - len(df)
)

print(
    "Removed duplicate records:",
    removed_duplicates
)

print(
    "Remaining:",
    len(df)
)


# ============================================================
# 8. VERIFY IMAGE FILES
# ============================================================

print("\nVerifying poster files...")

missing_images = []
valid_records = []

available_files = set(
    os.listdir(SOURCE_IMAGES)
)

# Build case-insensitive lookup
image_lookup = {
    os.path.splitext(filename)[0].lower(): filename
    for filename in available_files
    if filename.lower().endswith(".jpg")
}

for _, row in df.iterrows():

    movie_id = str(row["Id"]).strip()

    image_filename = image_lookup.get(
        movie_id.lower()
    )

    if image_filename is None:

        missing_images.append(
            movie_id
        )

    else:

        valid_records.append(
            (
                row,
                image_filename
            )
        )


print(
    "Valid poster matches:",
    len(valid_records)
)

print(
    "Missing posters:",
    len(missing_images)
)


# ============================================================
# 9. STOP IF IMAGES ARE MISSING
# ============================================================

if missing_images:

    print("\nWARNING:")
    print(
        "Some records do not have matching posters."
    )

    print(
        "\nFirst missing IDs:"
    )

    for movie_id in missing_images[:20]:
        print(" -", movie_id)

    print(
        "\nThe script will NOT create the final dataset "
        "until all images are verified."
    )

    raise SystemExit(
        "Missing poster files detected."
    )


# ============================================================
# 10. CREATE FINAL LABEL DATAFRAME
# ============================================================

print("\nCreating final label matrix...")

final_rows = []

for row, image_filename in valid_records:

    final_row = {
        "image_id": os.path.splitext(
            image_filename
        )[0]
    }

    for genre in TARGET_GENRES:

        final_row[genre] = int(
            row[genre]
        )

    final_rows.append(
        final_row
    )

final_df = pd.DataFrame(
    final_rows
)


# ============================================================
# 11. VERIFY LABELS
# ============================================================

print("\nVerifying final labels...")

final_df["_label_count"] = (
    final_df[TARGET_GENRES]
    .sum(axis=1)
)

zero_label_final = int(
    (
        final_df["_label_count"] == 0
    ).sum()
)

if zero_label_final > 0:

    raise SystemExit(
        "ERROR: Zero-label records found "
        "in final dataset."
    )

print(
    "PASS: Every final record has "
    "at least one target genre."
)


# ============================================================
# 12. VERIFY UNIQUE IMAGE IDS
# ============================================================

duplicate_images = int(
    final_df["image_id"]
    .duplicated()
    .sum()
)

print(
    "Duplicate image IDs:",
    duplicate_images
)

if duplicate_images > 0:

    raise SystemExit(
        "ERROR: Duplicate image IDs found."
    )


# ============================================================
# 13. COPY IMAGES
# ============================================================

print("\nCopying poster images...")

copied = 0
copy_errors = []

for row, image_filename in valid_records:

    source_path = os.path.join(
        SOURCE_IMAGES,
        image_filename
    )

    destination_path = os.path.join(
        OUTPUT_IMAGES,
        image_filename
    )

    try:

        if not os.path.exists(
            destination_path
        ):

            shutil.copy2(
                source_path,
                destination_path
            )

        copied += 1

    except Exception as e:

        copy_errors.append(
            (
                image_filename,
                str(e)
            )
        )

    if copied % 500 == 0:
        print(
            f"Copied {copied}/{len(valid_records)}"
        )


# ============================================================
# 14. CHECK COPY ERRORS
# ============================================================

if copy_errors:

    print("\nERRORS DURING IMAGE COPY:")

    for filename, error in copy_errors:
        print(
            filename,
            "->",
            error
        )

    raise SystemExit(
        "Image copying failed."
    )

print(
    "\nPASS:",
    copied,
    "images copied."
)


# ============================================================
# 15. SAVE LABEL CSV
# ============================================================

final_df = final_df.drop(
    columns=["_label_count"]
)

final_df.to_csv(
    OUTPUT_LABELS,
    index=False
)

print(
    "\nLabels saved to:"
)

print(
    OUTPUT_LABELS
)


# ============================================================
# 16. DATASET STATISTICS
# ============================================================

print("\nCalculating dataset statistics...")

total_records = len(final_df)

genre_counts = {}

for genre in TARGET_GENRES:

    genre_counts[genre] = int(
        final_df[genre].sum()
    )


# Number of labels per movie

label_counts = (
    final_df[TARGET_GENRES]
    .sum(axis=1)
)

exactly_one = int(
    (label_counts == 1).sum()
)

exactly_two = int(
    (label_counts == 2).sum()
)

exactly_three = int(
    (label_counts == 3).sum()
)

four_or_more = int(
    (label_counts >= 4).sum()
)


# ============================================================
# 17. WRITE DATASET REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "CINEVISION AI — MASTER DATASET REPORT\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    report.write(
        "SOURCE DATASET\n"
    )

    report.write(
        f"{SOURCE_DIR}\n\n"
    )

    report.write(
        "TARGET GENRES\n"
    )

    for index, genre in enumerate(
        TARGET_GENRES,
        start=1
    ):

        report.write(
            f"{index}. {genre}\n"
        )

    report.write("\n")

    report.write(
        "DATASET CLEANING\n"
    )

    report.write(
        f"Original records: "
        f"{before_no_target}\n"
    )

    report.write(
        f"Removed zero-target records: "
        f"{removed_no_target}\n"
    )

    report.write(
        f"Removed duplicate records: "
        f"{removed_duplicates}\n"
    )

    report.write(
        f"Final records: "
        f"{total_records}\n"
    )

    report.write(
        f"Final images: "
        f"{copied}\n\n"
    )

    report.write(
        "GENRE DISTRIBUTION\n"
    )

    report.write(
        "-" * 50 + "\n"
    )

    for genre, count in genre_counts.items():

        percentage = (
            count / total_records
        ) * 100

        report.write(
            f"{genre:12} : "
            f"{count:5} "
            f"({percentage:.2f}%)\n"
        )

    report.write("\n")

    report.write(
        "MULTI-LABEL DISTRIBUTION\n"
    )

    report.write(
        "-" * 50 + "\n"
    )

    report.write(
        f"Exactly 1 genre : "
        f"{exactly_one}\n"
    )

    report.write(
        f"Exactly 2 genres: "
        f"{exactly_two}\n"
    )

    report.write(
        f"Exactly 3 genres: "
        f"{exactly_three}\n"
    )

    report.write(
        f"4+ genres      : "
        f"{four_or_more}\n"
    )

    report.write("\n")

    report.write(
        "DATASET STATUS\n"
    )

    report.write(
        "Images copied successfully: YES\n"
    )

    report.write(
        "Missing images: 0\n"
    )

    report.write(
        "Duplicate image IDs: 0\n"
    )

    report.write(
        "Zero-label records: 0\n"
    )


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nFinal dataset records:",
    total_records
)

print(
    "Final poster images:",
    copied
)

print(
    "\nDataset location:"
)

print(
    OUTPUT_DIR
)

print(
    "\nLabels:"
)

print(
    OUTPUT_LABELS
)

print(
    "\nReport:"
)

print(
    OUTPUT_REPORT
)

print(
    "\nOriginal datasets were NOT modified."
)
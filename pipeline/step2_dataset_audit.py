import os
import pandas as pd
from PIL import Image
from collections import Counter

# ============================================================
# CINEVISION AI — STEP 2 DATASET AUDIT
# READ-ONLY SCRIPT
# ============================================================

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
    if os.path.basename(os.path.dirname(__file__)) == "pipeline"
    else os.path.join(os.path.dirname(__file__), "..")
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "Multi_Label_dataset"
)

CSV_PATH = os.path.join(
    DATASET_DIR,
    "train.csv"
)

IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "Images"
)

# ------------------------------------------------------------
# FINAL 8 GENRES
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# HELPER
# ------------------------------------------------------------

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 1. CHECK PATHS
# ============================================================

print_section("1. PATH VERIFICATION")

print("Project root :", PROJECT_ROOT)
print("Dataset      :", DATASET_DIR)
print("CSV          :", CSV_PATH)
print("Images       :", IMAGE_DIR)

if not os.path.exists(CSV_PATH):
    print("\nERROR: train.csv was not found.")
    print("Expected:", CSV_PATH)
    raise SystemExit

if not os.path.exists(IMAGE_DIR):
    print("\nERROR: Images folder was not found.")
    print("Expected:", IMAGE_DIR)
    raise SystemExit

print("\nPASS: Required dataset paths exist.")


# ============================================================
# 2. LOAD CSV
# ============================================================

print_section("2. CSV INFORMATION")

df = pd.read_csv(CSV_PATH)

print("Rows    :", len(df))
print("Columns :", len(df.columns))

print("\nColumns:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:2}. {col}")


# ============================================================
# 3. VERIFY TARGET GENRES
# ============================================================

print_section("3. TARGET GENRE VERIFICATION")

missing_genres = [
    genre for genre in TARGET_GENRES
    if genre not in df.columns
]

available_genres = [
    genre for genre in TARGET_GENRES
    if genre in df.columns
]

print("Required genres:")
for genre in TARGET_GENRES:
    status = "FOUND" if genre in df.columns else "MISSING"
    print(f"{genre:12} : {status}")

if missing_genres:
    print("\nWARNING: Missing target genres:", missing_genres)
else:
    print("\nPASS: All 8 target genres are present.")


# ============================================================
# 4. GENRE COUNTS
# ============================================================

print_section("4. FINAL 8-GENRE DISTRIBUTION")

genre_counts = {}

for genre in TARGET_GENRES:
    if genre in df.columns:
        count = int(df[genre].sum())
        genre_counts[genre] = count

for genre, count in genre_counts.items():
    percentage = (count / len(df)) * 100
    print(
        f"{genre:12} : {count:5} movies "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 5. CLASS IMBALANCE
# ============================================================

print_section("5. CLASS BALANCE ANALYSIS")

counts = list(genre_counts.values())

if counts:
    max_count = max(counts)
    min_count = min(counts)

    max_genre = max(genre_counts, key=genre_counts.get)
    min_genre = min(genre_counts, key=genre_counts.get)

    imbalance_ratio = max_count / min_count if min_count else 0

    print(
        f"Most frequent : {max_genre} ({max_count})"
    )

    print(
        f"Least frequent: {min_genre} ({min_count})"
    )

    print(
        f"Max/Min ratio : {imbalance_ratio:.2f}"
    )

    if imbalance_ratio <= 2:
        print("\nClass balance: GOOD")
    elif imbalance_ratio <= 4:
        print("\nClass balance: MODERATE")
    else:
        print("\nClass balance: IMBALANCED")


# ============================================================
# 6. MULTI-LABEL STATISTICS
# ============================================================

print_section("6. MULTI-LABEL STATISTICS")

existing_target_genres = [
    g for g in TARGET_GENRES if g in df.columns
]

df["_target_label_count"] = df[existing_target_genres].sum(axis=1)

label_distribution = (
    df["_target_label_count"]
    .value_counts()
    .sort_index()
)

print("Number of selected genres per movie:\n")

for label_count, movie_count in label_distribution.items():
    percentage = (movie_count / len(df)) * 100

    if label_count == 1:
        description = "Exactly 1 genre"
    elif label_count == 2:
        description = "Exactly 2 genres"
    elif label_count == 3:
        description = "Exactly 3 genres"
    elif label_count >= 4:
        description = "4+ genres"
    else:
        description = "No target genre"

    print(
        f"{description:20} : "
        f"{movie_count:5} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 7. MOVIES WITH NO TARGET GENRE
# ============================================================

print_section("7. MOVIES WITH NO TARGET GENRE")

zero_label_count = int(
    (df["_target_label_count"] == 0).sum()
)

print(
    "Movies with none of the 8 target genres:",
    zero_label_count
)

if zero_label_count == 0:
    print("PASS: Every movie has at least one target genre.")
else:
    print(
        "WARNING: These movies cannot be directly used "
        "for the final 8-class model."
    )


# ============================================================
# 8. ID DUPLICATES
# ============================================================

print_section("8. MOVIE ID DUPLICATE CHECK")

if "Id" in df.columns:

    duplicate_ids = int(
        df["Id"].duplicated().sum()
    )

    unique_ids = df["Id"].nunique()

    print("Total IDs :", len(df))
    print("Unique IDs:", unique_ids)
    print("Duplicate :", duplicate_ids)

    if duplicate_ids == 0:
        print("\nPASS: No duplicate movie IDs.")
    else:
        print("\nWARNING: Duplicate movie IDs found.")

else:
    print("WARNING: 'Id' column not found.")


# ============================================================
# 9. IMAGE FILE INVENTORY
# ============================================================

print_section("9. IMAGE FILE INVENTORY")

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

image_files = []

for filename in os.listdir(IMAGE_DIR):

    filepath = os.path.join(
        IMAGE_DIR,
        filename
    )

    if os.path.isfile(filepath):

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in valid_extensions:
            image_files.append(filename)

print("Total image files:", len(image_files))

extension_counts = Counter(
    os.path.splitext(x)[1].lower()
    for x in image_files
)

print("\nImage extensions:")

for extension, count in sorted(
    extension_counts.items()
):
    print(
        f"{extension:6} : {count}"
    )


# ============================================================
# 10. CSV → IMAGE ALIGNMENT
# ============================================================

print_section("10. CSV → IMAGE ALIGNMENT")

image_set = {
    os.path.splitext(filename)[0].lower()
    for filename in image_files
}

missing_images = []
matched_images = []

if "Id" in df.columns:

    for movie_id in df["Id"].astype(str):

        movie_id_clean = (
            movie_id.strip()
            .lower()
        )

        if movie_id_clean in image_set:
            matched_images.append(movie_id)
        else:
            missing_images.append(movie_id)

    print(
        "CSV records       :",
        len(df)
    )

    print(
        "Matched images    :",
        len(matched_images)
    )

    print(
        "Missing images    :",
        len(missing_images)
    )

    alignment_percentage = (
        len(matched_images) / len(df)
    ) * 100

    print(
        f"Alignment         : "
        f"{alignment_percentage:.2f}%"
    )

    if not missing_images:
        print(
            "\nPASS: Every CSV movie has "
            "a corresponding image."
        )

    else:
        print(
            "\nWARNING: Missing poster images found."
        )

else:
    print(
        "Cannot perform alignment check "
        "because Id column is missing."
    )


# ============================================================
# 11. IMAGE INTEGRITY CHECK
# ============================================================

print_section("11. IMAGE INTEGRITY CHECK")

corrupt_images = []
valid_images = 0

print(
    "Checking",
    len(image_files),
    "images..."
)

for index, filename in enumerate(
    image_files,
    start=1
):

    filepath = os.path.join(
        IMAGE_DIR,
        filename
    )

    try:

        with Image.open(filepath) as img:
            img.verify()

        valid_images += 1

    except Exception:
        corrupt_images.append(filename)

    if index % 500 == 0:
        print(
            f"Checked {index}/{len(image_files)}"
        )

print("\nValid images :", valid_images)
print("Corrupt images:", len(corrupt_images))

if not corrupt_images:
    print("\nPASS: No corrupt images detected.")
else:
    print("\nWARNING: Corrupt images detected.")


# ============================================================
# 12. IMAGE DIMENSIONS
# ============================================================

print_section("12. IMAGE DIMENSION ANALYSIS")

widths = []
heights = []

for filename in image_files:

    filepath = os.path.join(
        IMAGE_DIR,
        filename
    )

    try:

        with Image.open(filepath) as img:
            width, height = img.size

        widths.append(width)
        heights.append(height)

    except Exception:
        pass

if widths and heights:

    print(
        f"Average width : "
        f"{sum(widths) / len(widths):.1f}px"
    )

    print(
        f"Average height: "
        f"{sum(heights) / len(heights):.1f}px"
    )

    print(
        f"Minimum size  : "
        f"{min(widths)} × {min(heights)}"
    )

    print(
        f"Maximum width : "
        f"{max(widths)}px"
    )

    print(
        f"Maximum height: "
        f"{max(heights)}px"
    )


# ============================================================
# 13. EXTRA IMAGES NOT PRESENT IN CSV
# ============================================================

print_section("13. EXTRA IMAGE CHECK")

if "Id" in df.columns:

    csv_ids = {
        str(x).strip().lower()
        for x in df["Id"]
    }

    extra_images = []

    for filename in image_files:

        image_id = os.path.splitext(
            filename
        )[0].strip().lower()

        if image_id not in csv_ids:
            extra_images.append(filename)

    print(
        "Images not referenced by CSV:",
        len(extra_images)
    )

    if not extra_images:
        print(
            "PASS: No unreferenced images."
        )


# ============================================================
# 14. FINAL LABEL MATRIX PREVIEW
# ============================================================

print_section("14. LABEL MATRIX PREVIEW")

preview_columns = (
    ["Id"]
    + existing_target_genres
)

print(
    df[preview_columns]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 15. FINAL VERDICT
# ============================================================

print_section("15. CINEVISION AI STEP 2 VERDICT")

all_genres_exist = len(missing_genres) == 0

no_duplicate_ids = (
    duplicate_ids == 0
    if "Id" in df.columns
    else False
)

all_images_aligned = (
    len(missing_images) == 0
    if "Id" in df.columns
    else False
)

all_images_valid = (
    len(corrupt_images) == 0
)

has_valid_labels = (
    zero_label_count == 0
)

print(
    f"8 target genres present : "
    f"{'PASS' if all_genres_exist else 'FAIL'}"
)

print(
    f"No duplicate IDs        : "
    f"{'PASS' if no_duplicate_ids else 'FAIL'}"
)

print(
    f"All images aligned      : "
    f"{'PASS' if all_images_aligned else 'FAIL'}"
)

print(
    f"Image integrity         : "
    f"{'PASS' if all_images_valid else 'FAIL'}"
)

print(
    f"Every movie has label   : "
    f"{'PASS' if has_valid_labels else 'FAIL'}"
)

if (
    all_genres_exist
    and no_duplicate_ids
    and all_images_aligned
    and all_images_valid
    and has_valid_labels
):

    print("\n" + "=" * 70)
    print("STEP 2 PRIMARY DATASET: READY")
    print("=" * 70)

    print(
        "\nMulti_Label_dataset is suitable "
        "as the primary CineVision AI dataset."
    )

else:

    print("\n" + "=" * 70)
    print("STEP 2: REQUIRES CLEANUP / INVESTIGATION")
    print("=" * 70)


# ============================================================
# CLEAN TEMPORARY COLUMN
# ============================================================

df.drop(
    columns=["_target_label_count"],
    inplace=True
)

print("\nAudit completed.")
print("No dataset files were modified.")
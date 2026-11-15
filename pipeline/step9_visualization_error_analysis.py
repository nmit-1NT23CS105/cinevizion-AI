# ============================================================
# CINEVISION AI
# STEP 9 — VISUALIZATION & ERROR ANALYSIS
# ============================================================
#
# This script:
#   1. Plots training/validation curves
#   2. Plots per-genre metrics
#   3. Generates ROC curves
#   4. Generates Precision-Recall curves
#   5. Generates per-genre confusion matrices
#   6. Performs test-set error analysis
#   7. Identifies difficult predictions
#   8. Generates an automatic analysis report
#
# IMPORTANT:
#   - No model retraining
#   - No threshold optimization
#   - No modification of Steps 1–8
#   - Test thresholds come ONLY from validation
#
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    auc
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

from pathlib import Path

BASE_DIR = str(
    Path(__file__).resolve().parent.parent
    if Path(__file__).resolve().parent.name == "pipeline"
    else Path(__file__).resolve().parent
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

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "splits"
)

PLOT_DIR = os.path.join(
    EVAL_DIR,
    "plots"
)

ERROR_DIR = os.path.join(
    EVAL_DIR,
    "error_analysis"
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)

os.makedirs(
    ERROR_DIR,
    exist_ok=True
)


# ============================================================
# TARGET GENRES
# ============================================================

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_plot(filename):
    """
    Save current matplotlib figure.
    """

    path = os.path.join(
        PLOT_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[SAVED] {path}"
    )


def load_csv(path):
    """
    Load CSV and verify existence.
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )

    return pd.read_csv(path)


def load_thresholds():

    path = os.path.join(
        EVAL_DIR,
        "optimal_thresholds.json"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing threshold file:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def get_threshold(
    thresholds,
    genre
):

    value = thresholds[genre]

    if isinstance(
        value,
        dict
    ):

        return float(
            value["threshold"]
        )

    return float(value)


def load_test_data():

    test_path = os.path.join(
        SPLIT_DIR,
        "test.csv"
    )

    predictions_path = os.path.join(
        EVAL_DIR,
        "test_predictions.csv"
    )

    test_df = load_csv(
        test_path
    )

    predictions_df = load_csv(
        predictions_path
    )

    # --------------------------------------------------------
    # Verify test columns
    # --------------------------------------------------------

    expected_test_columns = (
        ["image_id"] +
        GENRES
    )

    missing_test = [
        col
        for col in expected_test_columns
        if col not in test_df.columns
    ]

    if missing_test:

        raise ValueError(
            "Missing columns in test.csv: "
            + str(missing_test)
        )

    # --------------------------------------------------------
    # Verify prediction columns
    # --------------------------------------------------------

    expected_prediction_columns = (
        ["image_id"] +
        [
            f"{genre}_probability"
            for genre in GENRES
        ]
    )

    missing_predictions = [
        col
        for col in expected_prediction_columns
        if col not in predictions_df.columns
    ]

    if missing_predictions:

        raise ValueError(
            "Missing columns in test_predictions.csv: "
            + str(missing_predictions)
        )

    # --------------------------------------------------------
    # Verify IDs
    # --------------------------------------------------------

    test_ids = set(
        test_df["image_id"]
    )

    prediction_ids = set(
        predictions_df["image_id"]
    )

    if test_ids != prediction_ids:

        raise ValueError(
            "Test image IDs and prediction image IDs "
            "do not match."
        )

    # --------------------------------------------------------
    # Merge using image_id ONLY
    #
    # Ground truth:
    #   Action
    #
    # Probability:
    #   Action_probability
    # --------------------------------------------------------

    merged = test_df.merge(
        predictions_df,
        on="image_id",
        how="inner",
        validate="one_to_one"
    )

    if len(merged) != len(test_df):

        raise ValueError(
            "Some test records were lost during merge."
        )

    print(
        f"Loaded test records: {len(merged)}"
    )

    return merged


# ============================================================
# 2. TRAINING HISTORY
# ============================================================

def plot_training_history():

    print("\n" + "=" * 70)
    print("1. TRAINING HISTORY")
    print("=" * 70)

    feature_path = os.path.join(
        MODEL_DIR,
        "feature_training_history.csv"
    )

    finetune_path = os.path.join(
        MODEL_DIR,
        "finetune_training_history.csv"
    )

    feature = load_csv(
        feature_path
    )

    finetune = load_csv(
        finetune_path
    )

    print(
        f"Feature extraction epochs: "
        f"{len(feature)}"
    )

    print(
        f"Fine-tuning epochs: "
        f"{len(finetune)}"
    )

    # --------------------------------------------------------
    # Global epoch numbering
    # --------------------------------------------------------

    feature["GlobalEpoch"] = np.arange(
        1,
        len(feature) + 1
    )

    finetune["GlobalEpoch"] = np.arange(
        len(feature) + 1,
        len(feature) + len(finetune) + 1
    )

    combined = pd.concat(
        [
            feature,
            finetune
        ],
        ignore_index=True
    )

    # ========================================================
    # LOSS
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        combined["GlobalEpoch"],
        combined["loss"],
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        combined["GlobalEpoch"],
        combined["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.axvline(
        x=len(feature) + 0.5,
        linestyle="--",
        label="Fine-Tuning Start"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Binary Cross-Entropy Loss"
    )

    plt.title(
        "EfficientNetB0 Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "01_training_validation_loss.png"
    )

    # ========================================================
    # PR-AUC
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        combined["GlobalEpoch"],
        combined["pr_auc"],
        marker="o",
        label="Training PR-AUC"
    )

    plt.plot(
        combined["GlobalEpoch"],
        combined["val_pr_auc"],
        marker="o",
        label="Validation PR-AUC"
    )

    plt.axvline(
        x=len(feature) + 0.5,
        linestyle="--",
        label="Fine-Tuning Start"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "PR-AUC"
    )

    plt.title(
        "EfficientNetB0 Training and Validation PR-AUC"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "02_training_validation_pr_auc.png"
    )

    # ========================================================
    # ROC-AUC
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        combined["GlobalEpoch"],
        combined["auc"],
        marker="o",
        label="Training ROC-AUC"
    )

    plt.plot(
        combined["GlobalEpoch"],
        combined["val_auc"],
        marker="o",
        label="Validation ROC-AUC"
    )

    plt.axvline(
        x=len(feature) + 0.5,
        linestyle="--",
        label="Fine-Tuning Start"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "ROC-AUC"
    )

    plt.title(
        "EfficientNetB0 Training and Validation ROC-AUC"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "03_training_validation_roc_auc.png"
    )


# ============================================================
# 3. PER-GENRE PERFORMANCE
# ============================================================

def plot_genre_metrics():

    print("\n" + "=" * 70)
    print("2. PER-GENRE PERFORMANCE")
    print("=" * 70)

    metrics_path = os.path.join(
        EVAL_DIR,
        "test_metrics.csv"
    )

    metrics = load_csv(
        metrics_path
    )

    print(
        "\nMetric columns found:"
    )

    print(
        metrics.columns.tolist()
    )

    required = [
        "Genre",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC"
    ]

    missing = [
        col
        for col in required
        if col not in metrics.columns
    ]

    if missing:

        raise ValueError(
            "Missing metric columns: "
            + str(missing)
        )

    # ========================================================
    # F1
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        metrics["Genre"],
        metrics["F1"]
    )

    plt.xlabel(
        "Genre"
    )

    plt.ylabel(
        "F1 Score"
    )

    plt.title(
        "Test F1 Score by Genre"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=30
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "04_test_f1_by_genre.png"
    )

    # ========================================================
    # PRECISION
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        metrics["Genre"],
        metrics["Precision"]
    )

    plt.xlabel(
        "Genre"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Test Precision by Genre"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=30
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "05_test_precision_by_genre.png"
    )

    # ========================================================
    # RECALL
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        metrics["Genre"],
        metrics["Recall"]
    )

    plt.xlabel(
        "Genre"
    )

    plt.ylabel(
        "Recall"
    )

    plt.title(
        "Test Recall by Genre"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=30
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "06_test_recall_by_genre.png"
    )

    # ========================================================
    # PR-AUC
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        metrics["Genre"],
        metrics["PR_AUC"]
    )

    plt.xlabel(
        "Genre"
    )

    plt.ylabel(
        "PR-AUC"
    )

    plt.title(
        "Test PR-AUC by Genre"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=30
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "07_test_pr_auc_by_genre.png"
    )

    # ========================================================
    # ROC-AUC
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        metrics["Genre"],
        metrics["ROC_AUC"]
    )

    plt.xlabel(
        "Genre"
    )

    plt.ylabel(
        "ROC-AUC"
    )

    plt.title(
        "Test ROC-AUC by Genre"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=30
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "08_test_roc_auc_by_genre.png"
    )

    # ========================================================
    # PRECISION / RECALL / F1
    # ========================================================

    x = np.arange(
        len(metrics)
    )

    width = 0.25

    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        x - width,
        metrics["Precision"],
        width,
        label="Precision"
    )

    plt.bar(
        x,
        metrics["Recall"],
        width,
        label="Recall"
    )

    plt.bar(
        x + width,
        metrics["F1"],
        width,
        label="F1"
    )

    plt.xticks(
        x,
        metrics["Genre"],
        rotation=30
    )

    plt.xlabel(
        "Genre"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Precision, Recall and F1 by Genre"
    )

    plt.ylim(
        0,
        1
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "09_precision_recall_f1_comparison.png"
    )


# ============================================================
# 4. ROC CURVES
# ============================================================

def plot_roc_curves():

    print("\n" + "=" * 70)
    print("3. ROC CURVES")
    print("=" * 70)

    data = load_test_data()

    plt.figure(
        figsize=(11, 8)
    )

    for genre in GENRES:

        # TRUE LABEL
        y_true = data[
            genre
        ].astype(int).values

        # PREDICTED PROBABILITY
        y_score = data[
            f"{genre}_probability"
        ].astype(float).values

        fpr, tpr, _ = roc_curve(
            y_true,
            y_score
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{genre} "
                f"(AUC={roc_auc:.3f})"
            )
        )

    # Random classifier
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curves — Test Set"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "10_roc_curves_all_genres.png"
    )


# ============================================================
# 5. PRECISION-RECALL CURVES
# ============================================================

def plot_pr_curves():

    print("\n" + "=" * 70)
    print("4. PRECISION-RECALL CURVES")
    print("=" * 70)

    data = load_test_data()

    plt.figure(
        figsize=(11, 8)
    )

    for genre in GENRES:

        # TRUE LABEL
        y_true = data[
            genre
        ].astype(int).values

        # PREDICTED PROBABILITY
        y_score = data[
            f"{genre}_probability"
        ].astype(float).values

        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                y_score
            )
        )

        pr_auc = auc(
            recall,
            precision
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"{genre} "
                f"(AUC={pr_auc:.3f})"
            )
        )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curves — Test Set"
    )

    plt.legend(
        loc="lower left"
    )

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "11_precision_recall_curves_all_genres.png"
    )


# ============================================================
# 6. CONFUSION MATRICES
# ============================================================

def plot_confusion_matrices():

    print("\n" + "=" * 70)
    print("5. PER-GENRE CONFUSION MATRICES")
    print("=" * 70)

    data = load_test_data()

    thresholds = load_thresholds()

    for genre in GENRES:

        threshold = get_threshold(
            thresholds,
            genre
        )

        # TRUE LABEL
        y_true = data[
            genre
        ].astype(int).values

        # PROBABILITY
        y_probability = data[
            f"{genre}_probability"
        ].astype(float).values

        # BINARY PREDICTION
        y_pred = (
            y_probability >= threshold
        ).astype(int)

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        plt.figure(
            figsize=(6, 5)
        )

        plt.imshow(
            cm
        )

        plt.title(
            f"{genre} — Confusion Matrix\n"
            f"Threshold = {threshold:.2f}"
        )

        plt.xlabel(
            "Predicted Label"
        )

        plt.ylabel(
            "Actual Label"
        )

        plt.xticks(
            [0, 1],
            [
                "Negative",
                "Positive"
            ]
        )

        plt.yticks(
            [0, 1],
            [
                "Negative",
                "Positive"
            ]
        )

        matrix_values = [
            [tn, fp],
            [fn, tp]
        ]

        for i in range(2):

            for j in range(2):

                plt.text(
                    j,
                    i,
                    str(
                        matrix_values[i][j]
                    ),
                    ha="center",
                    va="center",
                    fontsize=14
                )

        plt.colorbar()

        filename = (
            "12_confusion_matrix_"
            f"{genre.lower()}.png"
        )

        save_plot(
            filename
        )


# ============================================================
# 7. ERROR ANALYSIS
# ============================================================

def create_error_analysis():

    print("\n" + "=" * 70)
    print("6. ERROR ANALYSIS")
    print("=" * 70)

    data = load_test_data()

    thresholds = load_thresholds()

    # --------------------------------------------------------
    # Generate binary predictions
    # --------------------------------------------------------

    for genre in GENRES:

        threshold = get_threshold(
            thresholds,
            genre
        )

        probability = data[
            f"{genre}_probability"
        ].astype(float)

        data[
            f"{genre}_binary"
        ] = (
            probability >= threshold
        ).astype(int)

    # --------------------------------------------------------
    # Generate prediction categories
    # --------------------------------------------------------

    analysis_rows = []

    for idx in range(
        len(data)
    ):

        true_labels = [
            genre
            for genre in GENRES
            if data.iloc[idx][genre] == 1
        ]

        predicted_labels = [
            genre
            for genre in GENRES
            if data.iloc[idx][
                f"{genre}_binary"
            ] == 1
        ]

        true_set = set(
            true_labels
        )

        predicted_set = set(
            predicted_labels
        )

        intersection = (
            true_set &
            predicted_set
        )

        # ----------------------------------------------------
        # Category
        # ----------------------------------------------------

        if true_set == predicted_set:

            category = (
                "Exact Match"
            )

        elif len(intersection) > 0:

            category = (
                "Partial Match"
            )

        elif len(predicted_set) == 0:

            category = (
                "All False Negative"
            )

        else:

            category = (
                "All False Positive"
            )

        analysis_rows.append({

            "image_id":
                data.iloc[idx][
                    "image_id"
                ],

            "true_labels":
                ", ".join(
                    true_labels
                ),

            "predicted_labels":
                ", ".join(
                    predicted_labels
                ),

            "num_true_labels":
                len(true_set),

            "num_predicted_labels":
                len(predicted_set),

            "correct_labels":
                len(intersection),

            "error_category":
                category
        })

    error_df = pd.DataFrame(
        analysis_rows
    )

    error_path = os.path.join(
        ERROR_DIR,
        "test_error_analysis.csv"
    )

    error_df.to_csv(
        error_path,
        index=False
    )

    print(
        f"[SAVED] {error_path}"
    )

    # ========================================================
    # ERROR CATEGORY DISTRIBUTION
    # ========================================================

    category_counts = (
        error_df[
            "error_category"
        ]
        .value_counts()
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        category_counts.index,
        category_counts.values
    )

    plt.xlabel(
        "Prediction Category"
    )

    plt.ylabel(
        "Number of Posters"
    )

    plt.title(
        "Test Prediction Categories"
    )

    plt.xticks(
        rotation=25
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "21_error_category_distribution.png"
    )

    # ========================================================
    # GENRE ERROR COUNTS
    # ========================================================

    genre_errors = []

    for genre in GENRES:

        y_true = data[
            genre
        ].astype(int).values

        y_pred = data[
            f"{genre}_binary"
        ].astype(int).values

        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                y_pred
            ).ravel()
        )

        genre_errors.append({

            "Genre":
                genre,

            "True_Positive":
                tp,

            "False_Positive":
                fp,

            "False_Negative":
                fn,

            "True_Negative":
                tn
        })

    genre_error_df = pd.DataFrame(
        genre_errors
    )

    genre_error_path = os.path.join(
        ERROR_DIR,
        "genre_error_counts.csv"
    )

    genre_error_df.to_csv(
        genre_error_path,
        index=False
    )

    print(
        f"[SAVED] {genre_error_path}"
    )

    # ========================================================
    # FP / FN CHART
    # ========================================================

    x = np.arange(
        len(GENRES)
    )

    width = 0.35

    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        x - width / 2,
        genre_error_df[
            "False_Positive"
        ],
        width,
        label="False Positives"
    )

    plt.bar(
        x + width / 2,
        genre_error_df[
            "False_Negative"
        ],
        width,
        label="False Negatives"
    )

    plt.xticks(
        x,
        GENRES,
        rotation=30
    )

    plt.xlabel(
        "Genre"
    )

    plt.ylabel(
        "Number of Errors"
    )

    plt.title(
        "False Positives and False Negatives by Genre"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "22_false_positive_negative_by_genre.png"
    )

    return (
        error_df,
        genre_error_df
    )


# ============================================================
# 8. DIFFICULT CASE ANALYSIS
# ============================================================

def analyze_difficult_cases():

    print("\n" + "=" * 70)
    print("7. DIFFICULT CASE ANALYSIS")
    print("=" * 70)

    data = load_test_data()

    thresholds = load_thresholds()

    # --------------------------------------------------------
    # Calculate distance from threshold
    #
    # Smaller distance =
    # more uncertain prediction
    # --------------------------------------------------------

    for genre in GENRES:

        threshold = get_threshold(
            thresholds,
            genre
        )

        probability = data[
            f"{genre}_probability"
        ].astype(float)

        data[
            f"{genre}_difficulty"
        ] = np.abs(
            probability -
            threshold
        )

    difficulty_columns = [
        f"{genre}_difficulty"
        for genre in GENRES
    ]

    data[
        "average_threshold_distance"
    ] = data[
        difficulty_columns
    ].mean(
        axis=1
    )

    # --------------------------------------------------------
    # 50 most uncertain posters
    # --------------------------------------------------------

    difficult = (
        data
        .sort_values(
            "average_threshold_distance",
            ascending=True
        )
        .head(50)
    )

    output_columns = (
        ["image_id"]
        +
        GENRES
        +
        [
            f"{genre}_probability"
            for genre in GENRES
        ]
        +
        [
            "average_threshold_distance"
        ]
    )

    difficult_output = difficult[
        output_columns
    ]

    output_path = os.path.join(
        ERROR_DIR,
        "50_most_difficult_predictions.csv"
    )

    difficult_output.to_csv(
        output_path,
        index=False
    )

    print(
        f"[SAVED] {output_path}"
    )


# ============================================================
# 9. AUTOMATIC ERROR ANALYSIS REPORT
# ============================================================

def create_report(
    error_df,
    genre_error_df
):

    print("\n" + "=" * 70)
    print("8. GENERATING ERROR ANALYSIS REPORT")
    print("=" * 70)

    metrics_path = os.path.join(
        EVAL_DIR,
        "test_metrics.csv"
    )

    metrics = load_csv(
        metrics_path
    )

    report_path = os.path.join(
        EVAL_DIR,
        "step9_error_analysis_report.txt"
    )

    # --------------------------------------------------------
    # Best / worst genre
    # --------------------------------------------------------

    best = metrics.loc[
        metrics["F1"].idxmax()
    ]

    worst = metrics.loc[
        metrics["F1"].idxmin()
    ]

    # --------------------------------------------------------
    # Load thresholds
    # --------------------------------------------------------

    thresholds = load_thresholds()

    # --------------------------------------------------------
    # Write report
    # --------------------------------------------------------

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CINEVISION AI\n"
        )

        f.write(
            "STEP 9 — VISUALIZATION "
            "AND ERROR ANALYSIS REPORT\n"
        )

        f.write(
            "=" * 70 +
            "\n\n"
        )

        # ====================================================
        # BEST
        # ====================================================

        f.write(
            "1. BEST PERFORMING GENRE\n"
        )

        f.write(
            "-" * 50 +
            "\n"
        )

        f.write(
            f"Genre       : "
            f"{best['Genre']}\n"
        )

        f.write(
            f"Precision   : "
            f"{best['Precision']:.4f}\n"
        )

        f.write(
            f"Recall      : "
            f"{best['Recall']:.4f}\n"
        )

        f.write(
            f"F1          : "
            f"{best['F1']:.4f}\n"
        )

        f.write(
            f"ROC-AUC     : "
            f"{best['ROC_AUC']:.4f}\n"
        )

        f.write(
            f"PR-AUC      : "
            f"{best['PR_AUC']:.4f}\n\n"
        )

        # ====================================================
        # WORST
        # ====================================================

        f.write(
            "2. WEAKEST PERFORMING GENRE\n"
        )

        f.write(
            "-" * 50 +
            "\n"
        )

        f.write(
            f"Genre       : "
            f"{worst['Genre']}\n"
        )

        f.write(
            f"Precision   : "
            f"{worst['Precision']:.4f}\n"
        )

        f.write(
            f"Recall      : "
            f"{worst['Recall']:.4f}\n"
        )

        f.write(
            f"F1          : "
            f"{worst['F1']:.4f}\n"
        )

        f.write(
            f"ROC-AUC     : "
            f"{worst['ROC_AUC']:.4f}\n"
        )

        f.write(
            f"PR-AUC      : "
            f"{worst['PR_AUC']:.4f}\n\n"
        )

        # ====================================================
        # GENRE RANKING
        # ====================================================

        f.write(
            "3. GENRE PERFORMANCE RANKING\n"
        )

        f.write(
            "-" * 50 +
            "\n"
        )

        ranking = metrics.sort_values(
            "F1",
            ascending=False
        )

        for rank, (
            _,
            row
        ) in enumerate(
            ranking.iterrows(),
            start=1
        ):

            f.write(
                f"{rank}. {row['Genre']}: "
                f"F1={row['F1']:.4f}, "
                f"Precision={row['Precision']:.4f}, "
                f"Recall={row['Recall']:.4f}, "
                f"ROC-AUC={row['ROC_AUC']:.4f}, "
                f"PR-AUC={row['PR_AUC']:.4f}\n"
            )

        f.write(
            "\n"
        )

        # ====================================================
        # ERROR CATEGORIES
        # ====================================================

        f.write(
            "4. ERROR CATEGORY DISTRIBUTION\n"
        )

        f.write(
            "-" * 50 +
            "\n"
        )

        counts = (
            error_df[
                "error_category"
            ]
            .value_counts()
        )

        total = len(
            error_df
        )

        for category, count in (
            counts.items()
        ):

            percentage = (
                count /
                total *
                100
            )

            f.write(
                f"{category}: "
                f"{count} "
                f"({percentage:.2f}%)\n"
            )

        f.write(
            "\n"
        )

        # ====================================================
        # FP / FN
        # ====================================================

        f.write(
            "5. FALSE POSITIVE / FALSE NEGATIVE ANALYSIS\n"
        )

        f.write(
            "-" * 50 +
            "\n"
        )

        for _, row in (
            genre_error_df.iterrows()
        ):

            f.write(
                f"{row['Genre']}: "
                f"TP={row['True_Positive']}, "
                f"FP={row['False_Positive']}, "
                f"FN={row['False_Negative']}, "
                f"TN={row['True_Negative']}\n"
            )

        f.write(
            "\n"
        )

        # ====================================================
        # GENRE INTERPRETATION
        # ====================================================

        f.write(
            "6. GENRE-SPECIFIC INTERPRETATION\n"
        )

        f.write(
            "-" * 50 +
            "\n"
        )

        f.write(
            "Drama and Comedy show relatively strong "
            "classification performance. Their F1 scores "
            "indicate that the model can identify these "
            "genres more reliably from poster-level "
            "visual information.\n\n"
        )

        f.write(
            "Action also demonstrates useful predictive "
            "performance, particularly in recall. "
            "This indicates that the model can identify "
            "many Action posters, although false positives "
            "remain.\n\n"
        )

        f.write(
            "Crime, Thriller and Horror are more challenging "
            "genres. Their visual characteristics can overlap "
            "with other genres, which makes fine-grained "
            "classification difficult.\n\n"
        )

        f.write(
            "Horror requires particular attention because "
            "a relatively strong ROC-AUC does not necessarily "
            "produce a similarly strong F1 score. This can "
            "occur when the model ranks examples reasonably "
            "well but probability thresholding is difficult "
            "under class imbalance.\n\n"
        )

        # ====================================================
        # MULTI-LABEL INTERPRETATION
        # ====================================================

        f.write(
            "7. MULTI-LABEL ERROR INTERPRETATION\n"
        )

        f.write(
            "-" * 50 +
            "\n"
        )

        f.write(
            "CineVision AI is a multi-label classification "
            "system. A movie poster can therefore have "
            "multiple correct genres simultaneously.\n\n"
        )

        f.write(
            "For example, a poster can legitimately belong "
            "to Drama, Romance and Thriller at the same time. "
            "Therefore, a partial match means that the model "
            "correctly identified at least one true genre "
            "while missing or adding other genres.\n\n"
        )

        # ====================================================
        # THRESHOLDS
        # ====================================================

        f.write(
            "8. VALIDATION-OPTIMIZED THRESHOLDS\n"
        )

        f.write(
            "-" * 50 +
            "\n"
        )

        f.write(
            "The genre-specific probability thresholds were "
            "optimized using the validation set only and "
            "were locked before evaluating the test set.\n\n"
        )

        for genre in GENRES:

            threshold = get_threshold(
                thresholds,
                genre
            )

            f.write(
                f"{genre}: "
                f"{threshold:.2f}\n"
            )

        f.write(
            "\n"
        )

        # ====================================================
        # EVALUATION INTEGRITY
        # ====================================================

        f.write(
            "9. EVALUATION INTEGRITY\n"
        )

        f.write(
            "-" * 50 +
            "\n"
        )

        f.write(
            "The test set was not used for model training "
            "or threshold optimization. The final test "
            "evaluation uses the EfficientNetB0 model and "
            "thresholds selected using validation data.\n\n"
        )

        # ====================================================
        # LIMITATIONS
        # ====================================================

        f.write(
            "10. MODEL LIMITATIONS\n"
        )

        f.write(
            "-" * 50 +
            "\n"
        )

        f.write(
            "1. The dataset contains imbalanced genre "
            "distributions.\n\n"
        )

        f.write(
            "2. Some genres have considerably fewer "
            "training examples than others.\n\n"
        )

        f.write(
            "3. Movie posters may share visual characteristics "
            "across multiple genres.\n\n"
        )

        f.write(
            "4. Metadata-derived labels may not always "
            "correspond to visually obvious poster content.\n\n"
        )

        f.write(
            "5. Multi-label classification is more difficult "
            "than single-label classification because multiple "
            "labels may be simultaneously correct.\n\n"
        )

        # ====================================================
        # CONCLUSION
        # ====================================================

        f.write(
            "11. OVERALL CONCLUSION\n"
        )

        f.write(
            "-" * 50 +
            "\n"
        )

        f.write(
            "The EfficientNetB0 transfer-learning model "
            "provides a meaningful solution for multi-label "
            "movie genre prediction from poster images. "
            "The model demonstrates useful ranking and "
            "classification performance across the eight "
            "target genres.\n\n"
        )

        f.write(
            "The visualization and error analysis provide "
            "additional evidence for understanding model "
            "strengths, weaknesses, false positives, false "
            "negatives and uncertain predictions.\n"
        )

    print(
        f"[SAVED] {report_path}"
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("\n")

    print(
        "=" * 70
    )

    print(
        "CINEVISION AI — STEP 9"
    )

    print(
        "VISUALIZATION & ERROR ANALYSIS"
    )

    print(
        "=" * 70
    )

    try:

        # ----------------------------------------------------
        # 1. Training history
        # ----------------------------------------------------

        plot_training_history()

        # ----------------------------------------------------
        # 2. Per-genre metrics
        # ----------------------------------------------------

        plot_genre_metrics()

        # ----------------------------------------------------
        # 3. ROC curves
        # ----------------------------------------------------

        plot_roc_curves()

        # ----------------------------------------------------
        # 4. Precision-Recall curves
        # ----------------------------------------------------

        plot_pr_curves()

        # ----------------------------------------------------
        # 5. Confusion matrices
        # ----------------------------------------------------

        plot_confusion_matrices()

        # ----------------------------------------------------
        # 6. Error analysis
        # ----------------------------------------------------

        (
            error_df,
            genre_error_df
        ) = create_error_analysis()

        # ----------------------------------------------------
        # 7. Difficult predictions
        # ----------------------------------------------------

        analyze_difficult_cases()

        # ----------------------------------------------------
        # 8. Report
        # ----------------------------------------------------

        create_report(
            error_df,
            genre_error_df
        )

        # ====================================================
        # SUCCESS
        # ====================================================

        print(
            "\n" +
            "=" * 70
        )

        print(
            "STEP 9 COMPLETED SUCCESSFULLY"
        )

        print(
            "=" * 70
        )

        print(
            "\nGenerated folders:"
        )

        print(
            f"\nPlots:"
            f"\n{PLOT_DIR}"
        )

        print(
            f"\nError Analysis:"
            f"\n{ERROR_DIR}"
        )

        print(
            f"\nReport:"
            f"\n{os.path.join(EVAL_DIR, 'step9_error_analysis_report.txt')}"
        )

        print(
            "\nAll Step 9 visualizations and analysis "
            "have been generated successfully."
        )

    except Exception as e:

        print(
            "\n" +
            "=" * 70
        )

        print(
            "STEP 9 FAILED"
        )

        print(
            "=" * 70
        )

        print(
            f"\nError: {e}"
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
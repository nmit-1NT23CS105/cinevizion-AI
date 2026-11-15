import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# Optional imports for rich interactive charts with safe fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Import CineVision 2.0 SOTA Utilities
from utils.aesthetics import extract_color_palette, analyze_visual_mood
from utils.similarity import extract_poster_embedding, find_similar_posters
from utils.explainability import create_aesthetic_radar_chart

# Set dark background style for matplotlib fallbacks
plt.style.use("dark_background")

# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

GENRES = [
    "Action",
    "Adventure",
    "Comedy",
    "Crime",
    "Drama",
    "Horror",
    "Romance",
    "Thriller",
]

IMAGE_SIZE = (224, 224)

# Paths to models (2-Model Architecture Comparison)
CNN_MODEL_PATH = (
    PROJECT_ROOT / "models" / "baseline_cnn" / "baseline_cnn_best.keras"
)
EFF_MODEL_PATH = (
    PROJECT_ROOT / "models" / "efficientnet" / "efficientnet_best.keras"
)

# Paths to datasets and evaluation
THRESHOLD_PATH = PROJECT_ROOT / "evaluation" / "optimal_thresholds.json"
THRESHOLD_V2_PATH = PROJECT_ROOT / "evaluation" / "optimal_thresholds_v2.json"
MASTER_LABELS_PATH = PROJECT_ROOT / "dataset" / "master" / "labels.csv"
MASTER_IMAGES_DIR = PROJECT_ROOT / "dataset" / "master" / "images"
TEST_SPLIT_PATH = PROJECT_ROOT / "dataset" / "splits" / "test.csv"

TEST_METRICS_PATH = PROJECT_ROOT / "evaluation" / "test_metrics.csv"
TEST_PREDS_PATH = PROJECT_ROOT / "evaluation" / "test_predictions.csv"
VAL_METRICS_PATH = PROJECT_ROOT / "evaluation" / "validation_metrics.csv"

CNN_HISTORY_PATH = (
    PROJECT_ROOT / "models" / "baseline_cnn" / "training_history.csv"
)
EFF_FEAT_HISTORY_PATH = (
    PROJECT_ROOT / "models" / "efficientnet" / "feature_training_history.csv"
)
EFF_FINE_HISTORY_PATH = (
    PROJECT_ROOT / "models" / "efficientnet" / "finetune_training_history.csv"
)

ERROR_COUNTS_PATH = (
    PROJECT_ROOT / "evaluation" / "error_analysis" / "genre_error_counts.csv"
)
DIFFICULT_50_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "error_analysis"
    / "50_most_difficult_predictions.csv"
)
TEST_ERROR_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "error_analysis"
    / "test_error_analysis.csv"
)

PLOTS_DIR = PROJECT_ROOT / "evaluation" / "plots"

# ============================================================
# STREAMLIT PAGE CONFIG & CUSTOM CSS
# ============================================================

st.set_page_config(
    page_title="🎬 CineVision AI — Multi-Label Genre Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CINEMA NOIR GLASSMORPHIC DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

        /* Root Variables */
        :root {
            --bg-void: #06090e;
            --bg-surface: #0e131f;
            --card-glass: rgba(17, 24, 39, 0.72);
            --card-border: rgba(255, 255, 255, 0.08);
            --card-border-hover: rgba(99, 102, 241, 0.4);
            --indigo-glow: rgba(99, 102, 241, 0.25);
            --cyan-glow: rgba(6, 182, 212, 0.25);
        }

        /* Base Page Styling */
        .stApp {
            background: radial-gradient(circle at 50% 0%, #111827 0%, #06090e 100%);
            color: #f1f5f9;
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
        }

        /* Smooth Typography */
        h1, h2, h3, h4, .cinema-title {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.5px;
        }

        /* Hero Brand Banner */
        .hero-banner {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px 30px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(16px);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }

        .hero-title-group h1 {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .hero-title-group p {
            color: #94a3b8;
            font-size: 1.05rem;
            margin: 6px 0 0 0;
            font-weight: 400;
        }

        .hero-pills {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.12);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #e2e8f0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }

        .status-pill.active {
            border-color: #3b82f6;
            color: #60a5fa;
            background: rgba(37, 99, 235, 0.15);
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 8px #10b981;
        }

        /* Glass Cards */
        .glass-card {
            background: var(--card-glass);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 18px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(12px);
            transition: all 0.25s ease;
        }

        .glass-card:hover {
            border-color: var(--card-border-hover);
            transform: translateY(-2px);
            box-shadow: 0 12px 30px var(--indigo-glow);
        }

        /* Metric Tiles */
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.7) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px 20px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.4);
        }

        .metric-card-val {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
        }

        .metric-card-lbl {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #94a3b8;
            font-weight: 600;
        }

        .metric-card-sub {
            font-size: 0.75rem;
            color: #10b981;
            font-weight: 600;
            margin-top: 4px;
        }

        /* Genre Badges with Dynamic Themes */
        .genre-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 30px;
            font-size: 0.9rem;
            font-weight: 700;
            margin: 4px 6px 4px 0;
            transition: all 0.2s ease;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }

        .genre-pill.active {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.3) 0%, rgba(30, 64, 175, 0.4) 100%);
            border: 1px solid #3b82f6;
            color: #93c5fd;
        }

        .genre-pill.inactive {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.06);
            color: #64748b;
        }

        /* Modern Progress Bars */
        .prog-container {
            margin-bottom: 12px;
        }
        .prog-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .prog-bar-bg {
            background: rgba(30, 41, 59, 0.6);
            border-radius: 8px;
            height: 10px;
            overflow: hidden;
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .prog-bar-fill {
            height: 100%;
            border-radius: 8px;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .prog-bar-threshold {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            background-color: #ef4444;
            z-index: 2;
        }

        /* Custom Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: rgba(15, 23, 42, 0.6);
            padding: 8px;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            margin-bottom: 20px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 46px;
            border-radius: 10px;
            padding: 8px 18px;
            font-weight: 600;
            font-size: 0.95rem;
            color: #94a3b8;
            background-color: transparent;
            border: none;
            transition: all 0.2s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #f1f5f9;
            background: rgba(255, 255, 255, 0.04);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
        }

        /* Cinema Poster Card Frame */
        .poster-frame {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.25s ease;
        }
        .poster-frame:hover {
            transform: scale(1.02);
            box-shadow: 0 16px 40px rgba(59, 130, 246, 0.3);
        }

        /* Sidebar Styling */
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0b0f17 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            padding: 8px 20px;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #ffffff !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
        }

        /* Card & Badge Utilities */
        .dashboard-card {
            background: var(--card-glass);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 18px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(12px);
            transition: all 0.25s ease;
        }
        .dashboard-card:hover {
            border-color: var(--card-border-hover);
        }

        .metric-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #94a3b8;
            font-weight: 600;
            margin-top: 4px;
        }

        .genre-badge {
            display: inline-block;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.3) 0%, rgba(30, 64, 175, 0.4) 100%);
            border: 1px solid #3b82f6;
            color: #93c5fd;
            border-radius: 20px;
            padding: 5px 14px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 6px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }

        /* SOTA Highlight Pill */
        .badge-sota {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
            border: 1px solid #10b981;
            color: #34d399;
            border-radius: 12px;
            padding: 3px 10px;
            font-size: 0.8rem;
            font-weight: 700;
            display: inline-block;
        }

        /* Streamlit Dataframe / Table styling */
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CACHED DATA & MODEL LOADERS
# ============================================================


@st.cache_resource
def load_keras_model(model_type="eff"):

    target_path = EFF_MODEL_PATH if model_type == "eff" else CNN_MODEL_PATH

    if not target_path.exists():
        st.error(f"Model artifact missing at path: {target_path}")
        return None
    try:
        model = tf.keras.models.load_model(target_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model '{model_type}': {e}")
        return None


@st.cache_data
def load_optimal_thresholds(use_v2=False):

    target_path = THRESHOLD_V2_PATH if use_v2 and THRESHOLD_V2_PATH.exists() else THRESHOLD_PATH
    if not target_path.exists():
        st.error(f"Threshold artifact missing at path: {target_path}")
        return {g: 0.5 for g in GENRES}
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    thresholds = {}
    for genre in GENRES:
        val = data.get(genre, 0.5)
        if isinstance(val, dict):
            thresholds[genre] = float(
                val.get("threshold", val.get("optimal_threshold", 0.5))
            )
        else:
            thresholds[genre] = float(val)
    return thresholds


@st.cache_data
def load_csv_artifact(path_str):

    p = Path(path_str)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


LOCKED_THRESHOLDS_V1 = load_optimal_thresholds(use_v2=False)
LOCKED_THRESHOLDS_V2 = load_optimal_thresholds(use_v2=True)

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")

    arch_choice = st.radio(
        "Select Architecture",
        options=[
            "EfficientNetB0 (Model 2: Transfer Learning — Best Performing)",
            "Custom CNN (Model 1: Baseline from Scratch)",
        ],
        index=0,
        help="Guideline 9: Compare the Custom 4-Block CNN Baseline against Fine-Tuned EfficientNetB0 Transfer Learning.",
    )

    if "EfficientNetB0" in arch_choice:
        selected_model_key = "eff"
        is_eff = True
        st.success("🟢 **Model 2: EfficientNetB0** (Transfer Learning — Best Performing)")
    else:
        selected_model_key = "cnn"
        is_eff = False
        st.info("🟡 **Model 1: Custom 4-Block CNN** (Baseline Architecture)")

    st.divider()

    st.markdown("### 🎯 Decision Threshold")
    threshold_mode = st.radio(
        "Threshold Strategy",
        options=[
            "Optimal Per-Genre (V1)",
            "Joint Optimized (V2 SOTA)",
            "Default (0.50)",
            "Custom Global Threshold",
        ],
        index=1,
        help="V1 was tuned per-genre; V2 SOTA optimizes Macro F1 and tail-genre recall jointly.",
    )

    custom_thresh_val = 0.50
    if threshold_mode == "Custom Global Threshold":
        custom_thresh_val = st.slider(
            "Global Threshold Value",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
        )

    active_thresholds = {}
    for g in GENRES:
        if threshold_mode == "Optimal Per-Genre (V1)":
            active_thresholds[g] = LOCKED_THRESHOLDS_V1.get(g, 0.50)
        elif threshold_mode == "Joint Optimized (V2 SOTA)":
            active_thresholds[g] = LOCKED_THRESHOLDS_V2.get(g, 0.50)
        elif threshold_mode == "Default (0.50)":
            active_thresholds[g] = 0.50
        else:
            active_thresholds[g] = custom_thresh_val

    st.divider()

    st.markdown("### 🧠 Grad-CAM Overlay Settings")
    heatmap_alpha = st.slider(
        "Heatmap Overlay Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        help="Adjust the opacity of the Grad-CAM visualization overlay.",
    )

    st.divider()

    st.markdown("### ℹ️ System Specifications")
    st.markdown(
        """
        - **Input Resolution:** `224 × 224 RGB`
        - **Output Classes:** `8 Genres`
        - **Task:** `Multi-Label Classification`
        - **Framework:** `TensorFlow 2.16 / Keras 3`
        - **Final Architecture:** `EfficientNetB0`
        - **Pre-scaling:** `Internal [0, 255]`
        """
    )

model = load_keras_model(selected_model_key)

# ============================================================
# HERO BRAND BANNER
# ============================================================

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title-group">
            <h1>🎬 CineVision AI <span style="font-size:1rem; vertical-align:middle; background:linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; border:1px solid rgba(56, 189, 248, 0.4); border-radius:20px; padding:3px 12px; font-weight:700;">v2.0 SOTA</span></h1>
            <p>Cinema Noir Intelligence • Explainable Multi-Label Deep Learning & Visual Analytics</p>
        </div>
        <div class="hero-pills">
            <span class="status-pill active"><span class="pulse-dot"></span> Engine Online</span>
            <span class="status-pill">TensorFlow 2.16</span>
            <span class="status-pill">EfficientNetB0 SOTA</span>
            <span class="status-pill">8 Target Genres</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MAIN NAVIGATION TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🎬 Poster Classifier & Explainability",
        "📊 Exploratory Data Analysis",
        "📈 Model Performance & SOTA Benchmarks",
        "🔍 Reverse Visual Search",
        "🎨 'What-If' Art Director Studio",
        "🎓 Project Overview & Architecture",
    ]
)

# ============================================================
# PREPROCESSING & INFERENCE FUNCTIONS
# ============================================================


def preprocess_for_inference(img_pil, is_efficientnet=True):

    img_rgb = img_pil.convert("RGB")
    img_resized = img_rgb.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = np.asarray(img_resized, dtype=np.float32)

    if not is_efficientnet:
        img_array = img_array / 255.0

    img_batch = np.expand_dims(img_array, axis=0)
    return img_resized, img_batch


def generate_gradcam_heatmap(
    keras_model, img_batch, class_idx, is_efficientnet=True
):

    try:
        if is_efficientnet:
            eff_base = keras_model.get_layer("efficientnetb0")
            last_conv_layer = eff_base.get_layer("top_activation")

            eff_grad_model = tf.keras.models.Model(
                inputs=eff_base.inputs, outputs=last_conv_layer.output
            )

            with tf.GradientTape() as tape:
                conv_output = eff_grad_model(img_batch, training=False)
                tape.watch(conv_output)

                x = conv_output
                for layer in keras_model.layers:
                    if layer.name in ["poster_input", "efficientnetb0"]:
                        continue
                    x = layer(x)
                loss = x[:, class_idx]

            grads = tape.gradient(loss, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap /= max_val
            return heatmap.numpy()

        else:
            with tf.GradientTape() as tape:
                x = img_batch
                conv_output = None
                for layer in keras_model.layers:
                    x = layer(x)
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        conv_output = x
                tape.watch(conv_output)
                loss = x[:, class_idx]

            grads = tape.gradient(loss, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap /= max_val
            return heatmap.numpy()

    except Exception as e:
        return None


def create_heatmap_overlay(pil_img, heatmap_np, alpha=0.4):

    w, h = pil_img.size
    heatmap_resized = cv2.resize(heatmap_np, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    img_np = np.array(pil_img)
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
    return heatmap_color, overlay


# ============================================================
# TAB 1: POSTER CLASSIFIER & EXPLAINABILITY
# ============================================================

with tab1:
    st.markdown("### 🎬 Interactive Movie Poster Analysis")

    # 1-Click Iconic Demo Samples for effortless testing
    st.markdown("##### ⚡ 1-Click Iconic Demo Presets")
    demo_c1, demo_c2, demo_c3, demo_c4 = st.columns(4)
    with demo_c1:
        if st.button("🔥 Action: Mad Max", use_container_width=True, help="tt0086034: Action, Adventure, Thriller"):
            st.session_state["active_sample_id"] = "tt0086034"
            st.session_state["source_radio_idx"] = 1
    with demo_c2:
        if st.button("👻 Horror: Elm Street", use_container_width=True, help="tt0088117: Horror, Thriller"):
            st.session_state["active_sample_id"] = "tt0088117"
            st.session_state["source_radio_idx"] = 1
    with demo_c3:
        if st.button("🎭 Comedy: Trading Places", use_container_width=True, help="tt0085334: Comedy"):
            st.session_state["active_sample_id"] = "tt0085334"
            st.session_state["source_radio_idx"] = 1
    with demo_c4:
        if st.button("🎬 Drama: Right Stuff", use_container_width=True, help="tt0086510: Drama"):
            st.session_state["active_sample_id"] = "tt0086510"
            st.session_state["source_radio_idx"] = 1

    input_mode = st.radio(
        "Choose Image Input Source",
        options=[
            "Upload Custom Poster Image",
            "Select Test Dataset Sample",
        ],
        index=st.session_state.get("source_radio_idx", 1),
        horizontal=True,
    )

    selected_img_pil = None
    img_caption = ""
    ground_truth_genres = None

    if input_mode == "Upload Custom Poster Image":
        uploaded_file = st.file_uploader(
            "Upload Movie Poster (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload any movie poster image to perform instant multi-label genre classification.",
        )
        if uploaded_file is not None:
            selected_img_pil = Image.open(uploaded_file)
            img_caption = f"Uploaded Poster: {uploaded_file.name}"
            ground_truth_genres = "Unknown (User Upload)"
    else:
        test_df = load_csv_artifact(TEST_SPLIT_PATH)
        if not test_df.empty:
            sample_ids = test_df["image_id"].tolist()
            current_id = st.session_state.get("active_sample_id", "tt0086034")
            def_idx = sample_ids.index(current_id) if current_id in sample_ids else 0
            selected_id = st.selectbox(
                "Select a Movie Poster from Locked Test Dataset (661 samples)",
                options=sample_ids,
                index=def_idx,
                format_func=lambda x: f"Movie ID: {x}",
            )

            img_path = MASTER_IMAGES_DIR / f"{selected_id}.jpg"
            if img_path.exists():
                selected_img_pil = Image.open(img_path)
                img_caption = f"Test Sample: {selected_id}.jpg"

                row = test_df[test_df["image_id"] == selected_id].iloc[0]
                gt_list = [g for g in GENRES if row.get(g, 0) == 1]
                ground_truth_genres = (
                    ", ".join(gt_list) if gt_list else "None"
                )
            else:
                st.error(f"Image not found at path: {img_path}")
        else:
            st.warning("Test dataset CSV missing or empty.")

    if selected_img_pil is not None and model is not None:
        col_left, col_right = st.columns([1.1, 1.4], gap="large")

        img_resized, img_batch = preprocess_for_inference(
            selected_img_pil, is_efficientnet=is_eff
        )

        preds = model.predict(img_batch, verbose=0)[0]

        results = []
        for g, prob in zip(GENRES, preds):
            thr = active_thresholds[g]
            is_pred = float(prob) >= thr
            results.append(
                {
                    "Genre": g,
                    "Model Score": float(prob),
                    "Locked Threshold": thr,
                    "Predicted": is_pred,
                }
            )
        res_df = pd.DataFrame(results)

        with col_left:
            st.markdown("#### 🖼️ Input Poster Artwork")
            st.image(
                selected_img_pil,
                caption=img_caption,
                use_container_width=True,
            )
            st.markdown(
                f"<div style='color:#94a3b8; font-size:0.85rem; margin-top:-6px; margin-bottom:12px;'>📐 Original Size: <code>{selected_img_pil.width} × {selected_img_pil.height} px</code></div>",
                unsafe_allow_html=True,
            )

            if ground_truth_genres:
                if ground_truth_genres == "Unknown (User Upload)":
                    st.info(f"🏷️ **Ground Truth:** `{ground_truth_genres}`")
                else:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="padding:12px 18px; margin-bottom:14px; border-color:rgba(16,185,129,0.35);">
                            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.8px; color:#10b981; font-weight:700;">Ground Truth Reference</div>
                            <div style="font-size:1.05rem; font-weight:700; color:#f1f5f9; margin-top:2px;">{ground_truth_genres}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("#### 🎨 Dominant Palette & Mood")
            try:
                palette = extract_color_palette(selected_img_pil, n_colors=5)
                mood = analyze_visual_mood(selected_img_pil)

                swatch_html = '<div style="display:flex; height:28px; border-radius:8px; overflow:hidden; margin-bottom:10px; box-shadow:0 4px 12px rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.1);">'
                for c in palette:
                    swatch_html += f'<div style="background-color:{c["hex"]}; width:{c["percentage"]}%;" title="{c["hex"]} ({c["percentage"]}%)"></div>'
                swatch_html += '</div>'
                st.markdown(swatch_html, unsafe_allow_html=True)

                col_p_text = " &nbsp;•&nbsp; ".join([f"<code>{c['hex']}</code> ({c['percentage']}%)" for c in palette[:3]])
                st.markdown(f"<div style='font-size:0.8rem; color:#94a3b8; margin-bottom:8px;'>{col_p_text}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="display:flex; gap:8px; flex-wrap:wrap;">
                        <span class="badge-sota" style="background:rgba(59,130,246,0.15); border-color:#3b82f6; color:#93c5fd;">🌡️ {mood['temperature_tag']}</span>
                        <span class="badge-sota" style="background:rgba(168,85,247,0.15); border-color:#a855f7; color:#d8b4fe;">🎬 {mood['lighting_tag']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception:
                pass

        with col_right:
            st.markdown("#### 🎯 Detected Genre Signatures")

            pred_genres_df = res_df[res_df["Predicted"]]
            if not pred_genres_df.empty:
                badge_html = '<div style="margin-bottom:12px; display:flex; flex-wrap:wrap; gap:8px;">'
                for _, r in pred_genres_df.iterrows():
                    badge_html += f'<div class="genre-pill active">✓ {r["Genre"]} <span style="opacity:0.85; font-size:0.8rem;">({r["Model Score"]:.1%})</span></div>'
                badge_html += '</div>'
                st.markdown(badge_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="glass-card" style="padding:12px 16px; border-color:rgba(239,68,68,0.4); color:#fca5a5;">⚠️ No genre passed its active decision threshold.</div>',
                    unsafe_allow_html=True,
                )

            top_row = res_df.loc[res_df["Model Score"].idxmax()]
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-val" style="font-size:1.8rem;">{top_row["Genre"]}</div>
                        <div class="metric-card-lbl">Primary Visual Genre</div>
                        <div class="metric-card-sub">Top Predicted Class</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_m2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-val" style="font-size:1.8rem;">{top_row['Model Score']:.1%}</div>
                        <div class="metric-card-lbl">Peak Model Activation</div>
                        <div class="metric-card-sub">Sigmoid Output Score</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            st.markdown("#### ⚡ Real-Time Genre Confidence Meters")
            for _, r in res_df.sort_values(by="Model Score", ascending=False).iterrows():
                g_name = r["Genre"]
                sc = r["Model Score"]
                th = r["Locked Threshold"]
                is_hit = r["Predicted"]
                fill_color = "linear-gradient(90deg, #2563eb 0%, #38bdf8 100%)" if is_hit else "rgba(100, 116, 139, 0.45)"
                badge_status = f'<span style="color:#38bdf8; font-size:0.75rem; font-weight:700;">ACTIVE</span>' if is_hit else f'<span style="color:#64748b; font-size:0.75rem;">BELOW THR</span>'
                st.markdown(
                    f"""
                    <div class="prog-container">
                        <div class="prog-header">
                            <span><b>{g_name}</b> {badge_status}</span>
                            <span><b>{sc:.1%}</b> <small style="color:#94a3b8;">(thr: {th:.1%})</small></span>
                        </div>
                        <div class="prog-bar-bg">
                            <div class="prog-bar-fill" style="width: {min(sc*100, 100):.1f}%; background: {fill_color};"></div>
                            <div class="prog-bar-threshold" style="left: {min(th*100, 100):.1f}%;" title="Decision Threshold: {th:.1%}"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 🕸️ Aesthetic Multi-Genre Radar Footprint")
            fig_radar = create_aesthetic_radar_chart(
                GENRES,
                res_df["Model Score"].values,
                active_thresholds,
            )
            if fig_radar:
                fig_radar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9"),
                    height=280,
                    margin=dict(l=30, r=30, t=25, b=25),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()
        st.markdown("#### 📋 Detailed Prediction Breakdown")

        table_df = res_df.copy()
        table_df["Model Score"] = (table_df["Model Score"] * 100).map(
            "{:.2f}%".format
        )
        table_df["Active Threshold"] = (
            table_df["Locked Threshold"] * 100
        ).map("{:.1f}%".format)
        table_df["Prediction"] = table_df["Predicted"].map(
            {True: "✅ Positive", False: "—"}
        )

        st.dataframe(
            table_df[
                ["Genre", "Model Score", "Active Threshold", "Prediction"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.markdown("### 🧠 Explainability — Grad-CAM Visual Heatmaps")
        st.caption(
            "Grad-CAM highlights image regions that contributed most strongly to model genre predictions."
        )

        cam_mode = st.radio(
            "Saliency Mode",
            options=["Single Genre Heatmap", "Dual-Genre Comparative Heatmaps (Side-by-Side)"],
            horizontal=True,
        )

        if cam_mode == "Single Genre Heatmap":
            pred_genre_options = (
                pred_genres_df["Genre"].tolist()
                if not pred_genres_df.empty
                else GENRES
            )
            target_cam_genre = st.selectbox(
                "Select Genre for Visual Explanation",
                options=GENRES,
                index=GENRES.index(pred_genre_options[0]),
            )

            target_idx = GENRES.index(target_cam_genre)
            heatmap = generate_gradcam_heatmap(
                model, img_batch, target_idx, is_efficientnet=is_eff
            )

            if heatmap is not None:
                hm_color, overlay_img = create_heatmap_overlay(
                    selected_img_pil, heatmap, alpha=heatmap_alpha
                )

                c_cam1, c_cam2, c_cam3 = st.columns(3)
                with c_cam1:
                    st.markdown("**Original Poster**")
                    st.image(selected_img_pil, use_container_width=True)
                with c_cam2:
                    st.markdown(
                        f"**Grad-CAM Heatmap ({target_cam_genre})**"
                    )
                    st.image(hm_color, use_container_width=True)
                with c_cam3:
                    st.markdown(
                        f"**Heatmap Overlay (α = {heatmap_alpha:.2f})**"
                    )
                    st.image(overlay_img, use_container_width=True)
            else:
                st.info(
                    "Grad-CAM is unavailable for this specific model architecture execution."
                )
        else:
            col_dg1, col_dg2 = st.columns(2)
            with col_dg1:
                g_a = st.selectbox("Select First Genre", options=GENRES, index=0)
            with col_dg2:
                g_b = st.selectbox("Select Second Genre", options=GENRES, index=4)

            hm_a = generate_gradcam_heatmap(model, img_batch, GENRES.index(g_a), is_efficientnet=is_eff)
            hm_b = generate_gradcam_heatmap(model, img_batch, GENRES.index(g_b), is_efficientnet=is_eff)

            c_d1, c_d2, c_d3 = st.columns(3)
            with c_d1:
                st.markdown("**Original Poster**")
                st.image(selected_img_pil, use_container_width=True)
            with c_d2:
                st.markdown(f"**Visual Saliency: {g_a}**")
                if hm_a is not None:
                    _, ov_a = create_heatmap_overlay(selected_img_pil, hm_a, alpha=heatmap_alpha)
                    st.image(ov_a, use_container_width=True)
                else:
                    st.write("Unavailable")
            with c_d3:
                st.markdown(f"**Visual Saliency: {g_b}**")
                if hm_b is not None:
                    _, ov_b = create_heatmap_overlay(selected_img_pil, hm_b, alpha=heatmap_alpha)
                    st.image(ov_b, use_container_width=True)
                else:
                    st.write("Unavailable")

    elif model is None:
        st.error("Model failed to load. Please check artifact files.")
    else:
        st.info("👈 Upload a poster or select a test sample above to begin.")

# ============================================================
# TAB 2: EXPLORATORY DATA ANALYSIS
# ============================================================

with tab2:
    st.markdown("### 📊 Exploratory Data Analysis & Master Dataset Insight")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-val">6,638</div>
                <div class="metric-card-lbl">Master Posters</div>
                <div class="metric-card-sub">Curated Archive</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-val">8</div>
                <div class="metric-card-lbl">Target Genres</div>
                <div class="metric-card-sub">Multi-Label</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-val">5,317</div>
                <div class="metric-card-lbl">Training Set</div>
                <div class="metric-card-sub">80.1% Partition</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-val">660</div>
                <div class="metric-card-lbl">Validation Set</div>
                <div class="metric-card-sub">9.9% Partition</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-val">661</div>
                <div class="metric-card-lbl">Test Set</div>
                <div class="metric-card-sub">10.0% Locked</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    labels_df = load_csv_artifact(MASTER_LABELS_PATH)

    col_eda1, col_eda2 = st.columns(2, gap="large")

    with col_eda1:
        st.markdown("#### 🎭 Master Genre Frequency Distribution")
        if not labels_df.empty and set(GENRES).issubset(labels_df.columns):
            genre_counts = labels_df[GENRES].sum().reset_index()
            genre_counts.columns = ["Genre", "Count"]
            genre_counts = genre_counts.sort_values(
                by="Count", ascending=False
            )

            if HAS_PLOTLY:
                fig_genre_dist = px.bar(
                    genre_counts,
                    x="Genre",
                    y="Count",
                    text="Count",
                    color="Count",
                    color_continuous_scale=["#1e3a8a", "#3b82f6", "#38bdf8"],
                )
                fig_genre_dist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                    margin=dict(l=10, r=10, t=25, b=25),
                    height=360,
                )
                st.plotly_chart(fig_genre_dist, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.bar(
                    genre_counts["Genre"],
                    genre_counts["Count"],
                    color="#38bdf8",
                )
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.info("Master dataset labels unavailable.")

    with col_eda2:
        st.markdown("#### 🏷️ Multi-Label Target Distribution")
        if not labels_df.empty and set(GENRES).issubset(labels_df.columns):
            labels_df["num_genres"] = labels_df[GENRES].sum(axis=1)
            multi_label_counts = (
                labels_df["num_genres"].value_counts().reset_index()
            )
            multi_label_counts.columns = ["Genres per Movie", "Poster Count"]
            multi_label_counts = multi_label_counts.sort_values(
                by="Genres per Movie"
            )

            if HAS_PLOTLY:
                fig_multi = px.pie(
                    multi_label_counts,
                    values="Poster Count",
                    names="Genres per Movie",
                    hole=0.45,
                    color_discrete_sequence=["#3b82f6", "#818cf8", "#a855f7", "#ec4899", "#f59e0b"],
                )
                fig_multi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9"),
                    margin=dict(l=10, r=10, t=25, b=25),
                    height=360,
                )
                st.plotly_chart(fig_multi, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.bar(
                    multi_label_counts["Genres per Movie"].astype(str),
                    multi_label_counts["Poster Count"],
                    color="#ff8c42",
                )
                ax.set_xlabel("Genres per Movie")
                ax.set_ylabel("Poster Count")
                st.pyplot(fig)
        else:
            st.info("Multi-label statistics unavailable.")

    st.divider()

    st.markdown("#### 🖼️ Master Dataset Sample Posters Gallery")
    if not labels_df.empty:
        sample_rows = labels_df.sample(min(8, len(labels_df)), random_state=42)
        cols_gallery = st.columns(4)
        for i, (_, row) in enumerate(sample_rows.iterrows()):
            img_id = row["image_id"]
            path_g = MASTER_IMAGES_DIR / f"{img_id}.jpg"
            col_target = cols_gallery[i % 4]
            with col_target:
                if path_g.exists():
                    g_img = Image.open(path_g)
                    g_tags = [g for g in GENRES if row.get(g, 0) == 1]
                    st.image(
                        g_img,
                        use_container_width=True,
                    )
                    tags_pills = " ".join([f'<span class="badge-sota" style="font-size:0.7rem; padding:1px 6px;">{g}</span>' for g in g_tags])
                    st.markdown(
                        f"""
                        <div style="margin-top:-6px; margin-bottom:14px; font-size:0.85rem;">
                            <b>{img_id}</b><br>{tags_pills}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ============================================================
# TAB 3: MODEL PERFORMANCE & METRICS
# ============================================================

with tab3:
    st.markdown("### 📈 Comprehensive Model Performance & Evaluation")

    st.markdown("#### 🏆 Final Production Test Benchmarks (Model 2: EfficientNetB0)")
    tm1, tm2, tm3, tm4, tm5, tm6 = st.columns(6)
    with tm1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-val">0.5218</div>
                <div class="metric-card-lbl">Macro F1</div>
                <div class="metric-card-sub">+96.0% vs CNN</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tm2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-val">0.5834</div>
                <div class="metric-card-lbl">Micro F1</div>
                <div class="metric-card-sub">Harmonic Aggregate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tm3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-val">0.7840</div>
                <div class="metric-card-lbl">Macro ROC-AUC</div>
                <div class="metric-card-sub">+56.3% vs CNN</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tm4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-val">0.5104</div>
                <div class="metric-card-lbl">Macro PR-AUC</div>
                <div class="metric-card-sub">+120.1% vs CNN</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tm5:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-val">0.6849</div>
                <div class="metric-card-lbl">Macro Recall</div>
                <div class="metric-card-sub">+161.0% vs CNN</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tm6:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-val">93.19%</div>
                <div class="metric-card-lbl">Partial Match</div>
                <div class="metric-card-sub">≥1 Genre Correct</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='color:#94a3b8; font-size:0.8rem; margin-top:8px; margin-bottom:16px;'>* Exact Match Ratio strictly requires all 8 genre predictions to match ground truth simultaneously: <b>8.47%</b>. Partial Match (≥1 Genre Correct): <b>93.19%</b>.</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### ⚖️ Academic Guideline 9: Architecture & Performance Comparison")
    st.markdown(
        """
        <div class="glass-card" style="border-left: 4px solid #3b82f6;">
            <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; color:#38bdf8; font-weight:700; margin-bottom:6px;">Academic Requirement (Guideline 9)</div>
            <p style="margin-bottom:12px; font-size:0.95rem; color:#e2e8f0;">
                <i>"Compare two models, architectures, or parameter settings and justify the better-performing approach."</i>
            </p>
            <table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.9rem; color:#f1f5f9;">
                <thead>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color:#94a3b8;">
                        <th style="padding:10px 12px;">Evaluation Metric</th>
                        <th style="padding:10px 12px;">Model 1: Custom 4-Block CNN</th>
                        <th style="padding:10px 12px; color:#38bdf8;">Model 2: EfficientNetB0 (SOTA)</th>
                        <th style="padding:10px 12px; color:#10b981;">Delta / Improvement</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding:10px 12px; font-weight:600;">Macro F1</td>
                        <td style="padding:10px 12px;"><code>0.2662</code></td>
                        <td style="padding:10px 12px; font-weight:700; color:#38bdf8;"><code>0.5218</code></td>
                        <td style="padding:10px 12px;"><span class="badge-sota">+96.0% gain</span></td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding:10px 12px; font-weight:600;">Macro ROC-AUC</td>
                        <td style="padding:10px 12px;"><code>0.5015</code></td>
                        <td style="padding:10px 12px; font-weight:700; color:#38bdf8;"><code>0.7840</code></td>
                        <td style="padding:10px 12px;"><span class="badge-sota">+56.3% gain</span></td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding:10px 12px; font-weight:600;">Macro PR-AUC</td>
                        <td style="padding:10px 12px;"><code>0.2319</code></td>
                        <td style="padding:10px 12px; font-weight:700; color:#38bdf8;"><code>0.5104</code></td>
                        <td style="padding:10px 12px;"><span class="badge-sota">+120.1% gain</span></td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding:10px 12px; font-weight:600;">Macro Recall</td>
                        <td style="padding:10px 12px;"><code>0.2624</code></td>
                        <td style="padding:10px 12px; font-weight:700; color:#38bdf8;"><code>0.6849</code></td>
                        <td style="padding:10px 12px;"><span class="badge-sota">+161.0% gain</span></td>
                    </tr>
                    <tr>
                        <td style="padding:10px 12px; font-weight:600;">Partial Match (≥1 Hit)</td>
                        <td style="padding:10px 12px;"><code>82.51%</code></td>
                        <td style="padding:10px 12px; font-weight:700; color:#38bdf8;"><code>93.19%</code></td>
                        <td style="padding:10px 12px;"><span class="badge-sota">+10.68% abs</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **Key Technical Justification:**
        1. **Feature Hierarchy Depth:** The Custom CNN (trained from scratch on 5,317 samples) quickly overfits and suffers from *recall collapse* because random initialization cannot discern abstract movie artwork cues with limited training data.
        2. **Inductive Visual Priors:** EfficientNetB0 leverages pre-trained feature hierarchies (1.4 million ImageNet samples), transferring robust edge, texture, lighting, and semantic object priors directly into the movie poster domain.
        3. **Compound Scaling vs Flat Convolutions:** EfficientNet's inverted bottleneck MBConv blocks uniformly balance network depth, width, and resolution, capturing subtle multi-genre visual composition that flat convolution layers miss.
        4. **Parameter Setting Optimization:** Switching from the default global threshold (`0.50`) to Nelder-Mead Joint-Optimized Thresholds (`V2`) yields an additional jump in Macro F1 to **`0.5267`** and Partial Match to **`93.80%`**.
        """
    )

    st.divider()

    st.markdown(
        "#### 📉 Training History Curves (Feature Extraction vs Fine-Tuning)"
    )

    cnn_hist = load_csv_artifact(CNN_HISTORY_PATH)
    eff_feat_hist = load_csv_artifact(EFF_FEAT_HISTORY_PATH)
    eff_fine_hist = load_csv_artifact(EFF_FINE_HISTORY_PATH)

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        if HAS_PLOTLY:
            fig_loss = go.Figure()
            if not cnn_hist.empty and "val_loss" in cnn_hist.columns:
                fig_loss.add_trace(
                    go.Scatter(
                        y=cnn_hist["val_loss"],
                        mode="lines+markers",
                        name="CNN Val Loss",
                        line=dict(color="#f59e0b", width=2),
                    )
                )
            if not eff_feat_hist.empty and "val_loss" in eff_feat_hist.columns:
                fig_loss.add_trace(
                    go.Scatter(
                        y=eff_feat_hist["val_loss"],
                        mode="lines",
                        name="EffNet Phase A (Feature Ext)",
                        line=dict(color="#818cf8", width=2, dash="dash"),
                    )
                )
            if not eff_fine_hist.empty and "val_loss" in eff_fine_hist.columns:
                fig_loss.add_trace(
                    go.Scatter(
                        y=eff_fine_hist["val_loss"],
                        mode="lines+markers",
                        name="EffNet Phase B (Fine-Tune)",
                        line=dict(color="#38bdf8", width=2.5),
                    )
                )

            fig_loss.update_layout(
                title="Validation Loss over Epochs",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9"),
                xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Loss", gridcolor="rgba(255,255,255,0.06)"),
                height=320,
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            if not cnn_hist.empty and "val_loss" in cnn_hist.columns:
                ax.plot(cnn_hist["val_loss"], label="CNN Val Loss")
            if not eff_feat_hist.empty and "val_loss" in eff_feat_hist.columns:
                ax.plot(eff_feat_hist["val_loss"], label="EffNet Phase A")
            if not eff_fine_hist.empty and "val_loss" in eff_fine_hist.columns:
                ax.plot(eff_fine_hist["val_loss"], label="EffNet Phase B")
            ax.set_title("Validation Loss")
            ax.legend()
            st.pyplot(fig)

    with col_h2:
        if HAS_PLOTLY:
            fig_prauc = go.Figure()
            if not cnn_hist.empty and "val_pr_auc" in cnn_hist.columns:
                fig_prauc.add_trace(
                    go.Scatter(
                        y=cnn_hist["val_pr_auc"],
                        mode="lines+markers",
                        name="CNN Val PR-AUC",
                        line=dict(color="#f59e0b", width=2),
                    )
                )
            if (
                not eff_feat_hist.empty
                and "val_pr_auc" in eff_feat_hist.columns
            ):
                fig_prauc.add_trace(
                    go.Scatter(
                        y=eff_feat_hist["val_pr_auc"],
                        mode="lines",
                        name="EffNet Phase A (Feature Ext)",
                        line=dict(color="#818cf8", width=2, dash="dash"),
                    )
                )
            if (
                not eff_fine_hist.empty
                and "val_pr_auc" in eff_fine_hist.columns
            ):
                fig_prauc.add_trace(
                    go.Scatter(
                        y=eff_fine_hist["val_pr_auc"],
                        mode="lines+markers",
                        name="EffNet Phase B (Fine-Tune)",
                        line=dict(color="#38bdf8", width=2.5),
                    )
                )

            fig_prauc.update_layout(
                title="Validation PR-AUC over Epochs",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9"),
                xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="PR-AUC", gridcolor="rgba(255,255,255,0.06)"),
                height=320,
            )
            st.plotly_chart(fig_prauc, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            if not cnn_hist.empty and "val_pr_auc" in cnn_hist.columns:
                ax.plot(cnn_hist["val_pr_auc"], label="CNN Val PR-AUC")
            if (
                not eff_feat_hist.empty
                and "val_pr_auc" in eff_feat_hist.columns
            ):
                ax.plot(eff_feat_hist["val_pr_auc"], label="EffNet Phase A")
            if (
                not eff_fine_hist.empty
                and "val_pr_auc" in eff_fine_hist.columns
            ):
                ax.plot(eff_fine_hist["val_pr_auc"], label="EffNet Phase B")
            ax.set_title("Validation PR-AUC")
            ax.legend()
            st.pyplot(fig)

    st.divider()

    st.markdown("#### 📊 Per-Genre Performance Breakdown (Locked Test Set)")
    test_metrics_df = load_csv_artifact(TEST_METRICS_PATH)

    if not test_metrics_df.empty:
        st.dataframe(test_metrics_df, use_container_width=True, hide_index=True)

        if HAS_PLOTLY:
            fig_per_genre = px.bar(
                test_metrics_df,
                x="Genre",
                y=["Precision", "Recall", "F1", "PR_AUC"],
                barmode="group",
                title="Per-Genre Metric Comparison",
                color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#f59e0b"],
            )
            fig_per_genre.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                height=380,
            )
            st.plotly_chart(fig_per_genre, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            test_metrics_df.set_index("Genre")[
                ["Precision", "Recall", "F1", "PR_AUC"]
            ].plot(kind="bar", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    st.divider()

    st.markdown("#### 🖼️ Step 9 Evaluation Plots & Curves")
    plot_options = [
        "10_roc_curves_all_genres.png",
        "11_precision_recall_curves_all_genres.png",
        "09_precision_recall_f1_comparison.png",
        "21_error_category_distribution.png",
        "22_false_positive_negative_by_genre.png",
    ]
    selected_plot_file = st.selectbox(
        "Select Performance Plot to Display", options=plot_options
    )
    p_path = PLOTS_DIR / selected_plot_file
    if p_path.exists():
        st.image(Image.open(p_path), use_container_width=True)
    else:
        st.info(f"Plot missing: {selected_plot_file}")

    st.divider()

    st.markdown("#### 🔍 Deep Error Analysis & Difficult Predictions")

    col_err1, col_err2 = st.columns([1, 1.2])

    with col_err1:
        st.markdown("**Genre-wise Error Counts**")
        err_counts_df = load_csv_artifact(ERROR_COUNTS_PATH)
        if not err_counts_df.empty:
            st.dataframe(err_counts_df, use_container_width=True, hide_index=True)

    with col_err2:
        st.markdown("**50 Most Difficult Test Predictions**")
        diff_df = load_csv_artifact(DIFFICULT_50_PATH)
        if not diff_df.empty:
            selected_diff_id = st.selectbox(
                "Inspect Difficult Prediction Example",
                options=diff_df["image_id"].tolist(),
            )

            diff_row = diff_df[diff_df["image_id"] == selected_diff_id].iloc[0]
            diff_img_path = MASTER_IMAGES_DIR / f"{selected_diff_id}.jpg"

            if diff_img_path.exists():
                st.image(
                    Image.open(diff_img_path),
                    caption=f"Difficult Sample: {selected_diff_id}",
                    width=240,
                )

            st.write(f"**True Genres:** `{diff_row.get('true_genres', 'N/A')}`")
            st.write(
                f"**Predicted Genres:** `{diff_row.get('predicted_genres', 'N/A')}`"
            )
            st.write(
                f"**Loss / Error Score:** `{diff_row.get('loss', diff_row.get('error', 'N/A'))}`"
            )

# ============================================================
# TAB 4: REVERSE POSTER VISUAL SEARCH & RECOMMENDATIONS
# ============================================================

with tab4:
    st.markdown("### 🔍 Reverse Poster Visual Search & Recommendations")
    st.caption(
        "Using 1,280-dimensional deep feature embeddings from EfficientNetB0, CineVision AI searches through cinema history to find posters with matching artistic compositions, directorial styles, and lighting palettes."
    )

    col_vs_input, col_vs_results = st.columns([1, 2.5], gap="large")

    with col_vs_input:
        st.markdown("#### 🎯 Query Poster")
        vs_mode = st.radio("Source", ["Use Current Classifier Poster", "Upload Search Poster"], horizontal=True)

        query_pil = None
        if vs_mode == "Use Current Classifier Poster":
            query_pil = selected_img_pil
        else:
            search_up = st.file_uploader("Upload Poster for Visual Search", type=["jpg", "jpeg", "png", "webp"], key="search_uploader")
            if search_up is not None:
                query_pil = Image.open(search_up).convert("RGB")

        if query_pil:
            st.image(query_pil, caption="Query Artwork", use_container_width=True)
            top_k_val = st.slider("Number of Similar Movies to Retrieve", min_value=3, max_value=8, value=5)

    with col_vs_results:
        st.markdown("#### 🎬 Visually Similar Movies Found in Database")
        if query_pil is not None:
            with st.spinner("Extracting 1,280-d latent vector and querying cosine index..."):
                eff_model_for_search = load_keras_model("eff")
                if eff_model_for_search is not None:
                    q_emb = extract_poster_embedding(eff_model_for_search, query_pil)
                    similar_movies = find_similar_posters(q_emb, top_k=top_k_val)

                    if similar_movies:
                        cols = st.columns(len(similar_movies))
                        for idx, item in enumerate(similar_movies):
                            with cols[idx]:
                                if item["image_path"] and Path(item["image_path"]).exists():
                                    st.image(Image.open(item["image_path"]), use_container_width=True)
                                else:
                                    st.write("🖼️ [Image]")
                                sim_pct = item['similarity']
                                sim_color = "#10b981" if sim_pct >= 85 else ("#38bdf8" if sim_pct >= 75 else "#818cf8")
                                genres_str = ", ".join(item["genres"][:2])
                                st.markdown(
                                    f"""
                                    <div style="margin-top:-6px; margin-bottom:12px;">
                                        <span class="badge-sota" style="background:rgba(16,185,129,0.15); border-color:{sim_color}; color:{sim_color}; font-size:0.75rem;">
                                            ⚡ {sim_pct}% Match
                                        </span>
                                        <div style="font-size:0.8rem; font-weight:700; color:#f1f5f9; margin-top:4px;">{item['image_id']}</div>
                                        <div style="font-size:0.75rem; color:#94a3b8;">{genres_str}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    else:
                        st.info("Visual embeddings index is building or missing.")
        else:
            st.info("👈 Select or upload a poster to initiate visual vector search.")

# ============================================================
# TAB 5: 'WHAT-IF' POSTER ART DIRECTOR STUDIO
# ============================================================

with tab5:
    st.markdown("### 🎨 'What-If' Poster Art Director Studio")
    st.caption(
        "Simulate how visual artistic adjustments (lighting, contrast, saturation, vignette) alter deep learning model perception and predicted genre probabilities in real time."
    )

    col_studio_ctrl, col_studio_preview = st.columns([1, 2], gap="large")

    with col_studio_ctrl:
        st.markdown("#### 🎛️ Artistic Adjustments")
        st.markdown("<div class='glass-card' style='padding:16px 20px;'>", unsafe_allow_html=True)
        st_bright = st.slider("☀️ Brightness", 0.4, 1.8, 1.0, 0.05)
        st_contrast = st.slider("🌓 Contrast", 0.4, 2.0, 1.0, 0.05)
        st_sat = st.slider("🌈 Color Saturation", 0.0, 2.5, 1.0, 0.1)
        st_vignette = st.selectbox("🎬 Vignette Effect (Dark Borders)", ["None", "Subtle (Noir/Drama)", "Heavy (Horror/Thriller)"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.info("💡 *Tip:* Darken brightness and boost contrast to see how Drama shifts into Thriller/Horror!")

    with col_studio_preview:
        if selected_img_pil is not None:
            from PIL import ImageEnhance
            mod_img = selected_img_pil.copy()
            mod_img = ImageEnhance.Brightness(mod_img).enhance(st_bright)
            mod_img = ImageEnhance.Contrast(mod_img).enhance(st_contrast)
            mod_img = ImageEnhance.Color(mod_img).enhance(st_sat)

            if st_vignette != "None":
                w_v, h_v = mod_img.size
                arr_v = np.array(mod_img, dtype=np.float32)
                X = cv2.getGaussianKernel(w_v, int(w_v / 2))
                Y = cv2.getGaussianKernel(h_v, int(h_v / 2))
                kernel = Y * X.T
                mask = kernel / kernel.max()
                intensity = 0.5 if "Subtle" in st_vignette else 0.8
                mask = 1.0 - intensity * (1.0 - mask)
                arr_v = arr_v * mask[:, :, np.newaxis]
                mod_img = Image.fromarray(np.uint8(np.clip(arr_v, 0, 255)))

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("**Original Poster**")
                st.image(selected_img_pil, use_container_width=True)
            with col_p2:
                st.markdown("**Modified Artwork**")
                st.image(mod_img, use_container_width=True)

            if model is not None:
                _, b_orig = preprocess_for_inference(selected_img_pil, is_efficientnet=is_eff)
                _, b_mod = preprocess_for_inference(mod_img, is_efficientnet=is_eff)

                p_orig_vals = model.predict(b_orig, verbose=0)[0]
                p_mod_vals = model.predict(b_mod, verbose=0)[0]

                st.markdown("#### ⚡ AI Genre Probability Shift")

                if HAS_PLOTLY:
                    delta_vals = [pm - po for po, pm in zip(p_orig_vals, p_mod_vals)]
                    fig_shift = go.Figure()
                    fig_shift.add_trace(go.Bar(
                        y=GENRES,
                        x=[d * 100 for d in delta_vals],
                        orientation="h",
                        marker=dict(
                            color=["#10b981" if d >= 0 else "#ef4444" for d in delta_vals]
                        ),
                        text=[f"{d:+.1%}" for d in delta_vals],
                        textposition="auto"
                    ))
                    fig_shift.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#f1f5f9"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Δ Probability Shift (%)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                        height=260,
                        margin=dict(l=10, r=10, t=20, b=20)
                    )
                    st.plotly_chart(fig_shift, use_container_width=True)

                shift_data = []
                for g, po, pm in zip(GENRES, p_orig_vals, p_mod_vals):
                    delta = pm - po
                    shift_data.append({
                        "Genre": g,
                        "Original Confidence": f"{po:.1%}",
                        "Modified Confidence": f"{pm:.1%}",
                        "Delta": f"{delta:+.1%}",
                        "Shift Direction": "🔺 Up" if delta > 0.03 else ("🔻 Down" if delta < -0.03 else "— Stable")
                    })
                st.dataframe(pd.DataFrame(shift_data), use_container_width=True, hide_index=True)
        else:
            st.info("👈 Select or upload a poster to enter the Art Director Studio.")

# ============================================================
# TAB 6: PROJECT OVERVIEW & ARCHITECTURE
# ============================================================

with tab6:
    st.markdown("### 🎓 Educational Project Overview & System Architecture")

    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color:#38bdf8; margin-top:0;">📌 Problem Statement</h4>
            <p style="color:#e2e8f0; font-size:0.95rem; line-height:1.6; margin-bottom:0;">
                Movie posters frequently capture multiple overlapping narrative themes. Single-label genre classification is insufficient for real-world cinema recommendation systems. <b>CineVision AI</b> addresses this challenge by formulating movie poster analysis as a <b>Multi-Label Image Classification</b> deep learning problem.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color:#818cf8; margin-top:0;">🎯 Core Technical Objective</h4>
            <p style="color:#e2e8f0; font-size:0.95rem; line-height:1.6; margin-bottom:0;">
                Predict one or more target movie genres simultaneously from a raw poster image across <b>8 target classes</b>:
                <br><span class="badge-sota" style="margin-top:6px;">Action | Adventure | Comedy | Crime | Drama | Horror | Romance | Thriller</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### 🔄 End-to-End MLOps Pipeline Flowchart")
    st.code(
        """
    +------------------------+
    |   Master Dataset       | (6,638 posters, 8 target genres)
    +-----------+------------+
                |
                v
    +------------------------+
    | Multi-Label Stratified | (Train: 80.1%, Validation: 9.9%, Test: 10.0%)
    | Data Splitting         |
    +-----------+------------+
                |
                v
    +------------------------+
    | Preprocessing Pipeline | (224x224 RGB, float32, [0,255] internal scaling)
    +-----------+------------+
                |
                +---------------------------------------+
                |                                       |
                v                                       v
    +------------------------+             +------------------------+
    | Baseline CNN Model     |             | EfficientNetB0 (TL)    |
    | (4 Conv Blocks)        |             | (Phase A & B Fine-Tune)|
    +-----------+------------+             +-----------+------------+
                |                                       |
                +-------------------+-------------------+
                                    |
                                    v
                        +------------------------+
                        | Threshold Optimization | (Validation PR-AUC / F1)
                        +-----------+------------+
                                    |
                                    v
                        +------------------------+
                        | Locked Test Evaluation | (Macro F1: 0.5218, Grad-CAM)
                        +-----------+------------+
                                    |
                                    v
                        +------------------------+
                        | Interactive Streamlit  |
                        | Analytics Dashboard    |
                        +------------------------+
    """,
        language="text",
    )

    st.divider()

    col_ov1, col_ov2 = st.columns(2, gap="large")

    with col_ov1:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#38bdf8; margin-top:0;">🛠️ Technology Stack</h4>
                <ul style="color:#cbd5e1; font-size:0.9rem; line-height:1.7;">
                    <li><b>Language:</b> Python 3.10+</li>
                    <li><b>Deep Learning:</b> TensorFlow 2.16 / Keras 3</li>
                    <li><b>Transfer Learning:</b> EfficientNetB0 (ImageNet weights)</li>
                    <li><b>Data Science:</b> NumPy, Pandas, Scikit-Learn</li>
                    <li><b>Computer Vision:</b> OpenCV, Pillow (PIL)</li>
                    <li><b>Interactive Visualization:</b> Streamlit, Plotly, Matplotlib</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#a855f7; margin-top:0;">📁 Dataset Source Disclaimer</h4>
                <p style="color:#cbd5e1; font-size:0.9rem; margin-bottom:0;">
                    <b>Dataset source:</b> Public movie-poster dataset used for academic experimentation; source reference to be provided in the final project report.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ov2:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#10b981; margin-top:0;">👨‍💻 Individual Contribution Statement</h4>
                <p style="color:#cbd5e1; font-size:0.9rem; margin-bottom:0;">
                    The project was completed individually, including dataset preparation, preprocessing, model development, transfer learning, evaluation, visualization, error analysis, and deployment of the prediction application.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#f59e0b; margin-top:0;">⚠️ Key Project Limitations</h4>
                <ul style="color:#cbd5e1; font-size:0.9rem; line-height:1.7; margin-bottom:0;">
                    <li>Movie posters contain inherent visual ambiguity.</li>
                    <li>Significant class imbalance (e.g., Drama vs. Horror).</li>
                    <li>Sigmoid outputs are raw activations, not calibrated probabilities.</li>
                    <li>Exact Match Ratio is extremely stringent for multi-label tasks.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color:#38bdf8; margin-top:0;">🐳 Production Dockerization & Deployment</h4>
            <p style="color:#cbd5e1; font-size:0.9rem;">CineVision AI includes a self-contained, multi-stage <code>Dockerfile</code> and <code>requirements.txt</code> configured for one-click deployment on <b>Streamlit Community Cloud</b>, <b>HuggingFace Spaces</b>, or <b>Google Cloud Run</b>.</p>
            <pre style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; color: #38bdf8; font-family: monospace;">docker build -t cinevision-ai .
docker run -p 8501:8501 cinevision-ai</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-size: 0.9rem; padding: 14px 0;">
        <b>🎬 CineVision AI</b> — Multi-Label Movie Genre Classification & Visual Analytics<br>
        <span style="font-size: 0.8rem; color: #64748b;">Powered by TensorFlow 2.16, Keras 3 & Streamlit | Academic Deep Learning Pipeline</span>
    </div>
    """,
    unsafe_allow_html=True,
)
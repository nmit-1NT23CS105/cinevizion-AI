# 🎬 CineVision AI: Explainable Multi-Label Movie Poster Analytics & Recommendation Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%20%2F%20Keras%203-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Academic%20Research-green.svg)](LICENSE)

**CineVision AI** is an explainable, multi-label deep learning vision platform engineered to analyze movie poster artwork, predict cinematic genres across 8 categories, extract visual aesthetics and dominant color palettes, perform reverse visual vector search across cinema history, and simulate artistic adjustments in an interactive "What-If" Art Director Studio.

The platform is delivered as a production-grade, **Cinema Noir & Cyberpunk Elegance** interactive **Streamlit Web Application** featuring 6 specialized analytics and exploration tabs.

---

## 🌟 Key Capabilities & Architectural Highlights

1. **Multi-Label Genre Classification**:
   - Formulated as multi-label sigmoid classification across **8 target cinema genres**: *Action, Adventure, Comedy, Crime, Drama, Horror, Romance, Thriller*.
   - Evaluated on a locked 661-sample test split with per-genre threshold optimization.
2. **Academic Guideline 9 Architecture Comparison**:
   - **Model 1: Custom 4-Block CNN** (Trained from scratch with BatchNorm, Dropout, and unweighted BCE).
   - **Model 2: EfficientNetB0** (Transfer learning leveraging 1.4M ImageNet priors, compound MBConv scaling, and 2-phase fine-tuning).
3. **1-Click Iconic Demo Presets**:
   - Zero-click instant testing presets for iconic movie posters:
     - `🔥 Action: Mad Max` (`tt0086034`)
     - `👻 Horror: A Nightmare on Elm Street` (`tt0088117`)
     - `🎭 Comedy: Trading Places` (`tt0085334`)
     - `🎬 Drama: The Right Stuff` (`tt0086510`)
4. **Real-Time Confidence Meters**:
   - Custom horizontal gradient meters with dynamic threshold markers showing the exact decision boundary for each genre.
5. **Explainable AI (Grad-CAM & Dual-Genre Saliency)**:
   - Single-genre and side-by-side **Dual-Genre Grad-CAM heatmaps** revealing the spatial features that trigger multiple genres simultaneously on the same poster artwork.
6. **Aesthetic Color Palette & Visual Lighting Mood**:
   - K-Means ($K=5$) dominant color clustering with interactive hex swatches.
   - Temperature metric (Warm vs. Cool balance) and Lighting index (High-Key brightness vs. Low-Key Noir shadows).
7. **8-Axis Multi-Genre Aesthetic Radar Chart**:
   - Visualizes the full multi-dimensional narrative footprint of any movie poster against active decision thresholds.
8. **Reverse Poster Visual Search & Recommendations**:
   - Deep 1,280-dimensional latent feature vector extraction and cosine similarity search over 1,321 reference film posters.
   - Instantly retrieves the **Top 5 most visually similar movies in cinema history**.
9. **Interactive "What-If" Poster Art Director Studio**:
   - Real-time in-browser sliders for Brightness, Contrast, Saturation, and Vignette shading with live differential AI genre re-predictions and delta shift visualization.
10. **Production Containerization**:
    - Multi-stage `Dockerfile` and pinned `requirements.txt` configured for single-port (`8501`) deployment on Streamlit Community Cloud, HuggingFace Spaces, or Google Cloud Run.

---

## ⚖️ Model Comparison & Justification (Academic Guideline 9)

Evaluated strictly on the locked 661-sample test split:

| Evaluation Dimension | Model 1: Custom 4-Block CNN (From Scratch) | Model 2: EfficientNetB0 (Transfer Learning) | Relative Gain / Improvement |
| :--- | :---: | :---: | :---: |
| **Macro $F_1$** | `0.2662` | **`0.5218`** | **+96.0% gain** |
| **Micro $F_1$** | `0.3540` | **`0.5834`** | **+64.8% gain** |
| **Macro ROC-AUC** | `0.5015` | **`0.7840`** | **+56.3% gain** |
| **Macro PR-AUC** | `0.2319` | **`0.5104`** | **+120.1% gain** |
| **Macro Recall** | `0.2624` | **`0.6849`** | **+161.0% gain** |
| **Macro Precision** | `0.2701` | **`0.4412`** | **+63.3% gain** |
| **Exact Match Ratio** | `8.02%` | **`8.47%`** | **+0.45% absolute** |
| **Partial Match ($\ge 1$ Hit)** | `82.51%` | **`93.19%`** | **+10.68% absolute** |

### Key Technical Justifications:
1. **Feature Hierarchy Depth**: The Custom CNN (trained from scratch on 5,317 samples) quickly overfits and suffers from *recall collapse* because random initialization cannot discern abstract movie artwork cues with limited training data.
2. **Inductive Visual Priors**: EfficientNetB0 leverages pre-trained feature hierarchies (1.4 million ImageNet samples), transferring robust edge, texture, lighting, and semantic object priors directly into the movie poster domain.
3. **Compound Scaling vs. Flat Convolutions**: EfficientNet's inverted bottleneck MBConv blocks uniformly balance network depth, width, and resolution, capturing subtle multi-genre visual composition that flat convolution layers miss.
4. **Parameter Setting Optimization**: Switching from the default global threshold (`0.50`) to Nelder-Mead Joint-Optimized Thresholds (`V2`) yields an additional jump in Macro F1 to **`0.5267`** and Partial Match to **`93.80%`**.

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation
```bash
git clone https://github.com/nmit-1NT23CS105/cinevizion-AI.git
cd cinevizion-AI
pip install -r requirements.txt
```

### 2. Running the Interactive Streamlit Dashboard

**Option A: One-Click Windows Launcher**
Double-click:
```cmd
start_dashboard.bat
```

**Option B: Terminal Command**
```bash
streamlit run app.py
```
Opens in your browser at `http://localhost:8501`.

### 3. Running with Docker
```bash
docker build -t cinevision-ai .
docker run -p 8501:8501 cinevision-ai
```

---

## 📁 Project Architecture & Directory Structure

```
cinevizion-AI/
├── app.py                         # 6-Tab Streamlit interactive analytics platform
├── start_dashboard.bat            # 1-click Windows launcher for Streamlit
├── Dockerfile                     # Multi-stage production container
├── requirements.txt               # Pinned Python dependencies
├── README.md                      # Comprehensive project documentation
├── .gitignore                     # Git ignore rules for Python, caches, and models
├── .dockerignore                  # Docker ignore rules
│
├── utils/                         # Core platform utilities
│   ├── __init__.py                # Package initializer
│   ├── aesthetics.py              # K-Means color clustering & visual mood analysis
│   ├── similarity.py              # 1280-d latent vector search & recommendations
│   └── explainability.py          # Dual Grad-CAM & 8-axis aesthetic radar chart
│
├── pipeline/                      # Core Reproducible Machine Learning Pipeline
│   ├── __init__.py                # Pipeline package initializer
│   ├── step2_dataset_audit.py                  # Dataset integrity & resolution audit
│   ├── step3_create_master_dataset.py          # Master dataset assembly & verification
│   ├── step4_create_splits.py                  # Stratified train/val/test multi-label splits
│   ├── step5_preprocessing_pipeline.py         # Image preprocessing & data augmentation
│   ├── step6_baseline_cnn.py                   # Custom 4-block Baseline CNN training
│   ├── step7_efficientnet.py                   # EfficientNetB0 transfer learning & fine-tuning
│   ├── step8_evaluate_model.py                 # Multi-label evaluation & metrics computation
│   ├── step9_visualization_error_analysis.py   # Error analysis, confusion matrices & curves
│   ├── step11_joint_threshold_optimizer.py     # Nelder-Mead joint threshold optimization
│   └── step15_build_visual_index.py            # Latent visual embedding index generation
│
├── dataset/
│   ├── master/                    # 6,638 posters & labels.csv
│   │   ├── images/                # Poster artwork images (224x224 RGB)
│   │   └── labels.csv             # Multi-label ground truth annotations
│   └── splits/                    # train.csv, validation.csv, test.csv
│
├── models/
│   ├── baseline_cnn/              # Model 1: Custom 4-block CNN (from scratch)
│   │   ├── baseline_cnn_best.keras
│   │   ├── model_config.json
│   │   └── training_history.csv
│   └── efficientnet/              # Model 2: EfficientNetB0 (Transfer Learning — Production SOTA)
│       ├── efficientnet_best.keras
│       ├── feature_training_history.csv
│       ├── finetune_training_history.csv
│       ├── model_config.json
│       └── positive_weights.json
│
└── evaluation/
    ├── optimal_thresholds.json    # V1 per-genre thresholds
    ├── optimal_thresholds_v2.json # Joint-optimized V2 thresholds
    ├── test_metrics.csv           # Final per-genre test metrics
    ├── test_predictions.csv       # Test prediction probabilities
    ├── validation_metrics.csv     # Validation metrics
    ├── evaluation_summary.txt     # Complete evaluation report
    ├── visual_embeddings.npz      # Cached 1,280-d vector search database
    ├── plots/                     # 21 analytical plots & evaluation curves
    └── error_analysis/            # Error breakdowns & difficult sample logs
```

---

## ⚖️ Academic Disclaimers

- **Dataset Source**: Public movie-poster dataset used for academic experimentation; source reference to be provided in the final project report.
- **Individual Contribution**: The project was completed individually, including dataset preparation, preprocessing, baseline CNN and transfer learning models, threshold optimization, reverse vector search, Grad-CAM explainability, and web deployment.

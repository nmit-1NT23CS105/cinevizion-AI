import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# ============================================================
# CINEVISION AI - AESTHETICS & COLOR PALETTE UTILITY
# ============================================================

def rgb_to_hex(rgb):
    """Converts an RGB tuple (0-255) to a Hex color string."""
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

def extract_color_palette(pil_img, n_colors=5):
    """
    Extracts the top n_colors dominant colors from an image using K-Means clustering.
    Returns:
        list of dicts with 'rgb', 'hex', and 'percentage'.
    """
    # Downsample image for fast clustering
    img_small = pil_img.convert("RGB").resize((120, 180), Image.Resampling.LANCZOS)
    pixels = np.array(img_small).reshape(-1, 3).astype(float)

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=3)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    # Count occurrences
    counts = np.bincount(labels, minlength=n_colors)
    total = len(labels)
    percentages = counts / total

    # Sort descending by prevalence
    sort_indices = np.argsort(-percentages)

    palette = []
    for idx in sort_indices:
        rgb = tuple(int(round(c)) for c in centers[idx])
        palette.append({
            "rgb": rgb,
            "hex": rgb_to_hex(rgb),
            "percentage": round(float(percentages[idx]) * 100, 1)
        })

    return palette

def analyze_visual_mood(pil_img):
    """
    Analyzes visual lighting mood, contrast, and color temperature.
    Returns a dict with mood metrics and descriptive tags.
    """
    img_rgb = pil_img.convert("RGB").resize((120, 180), Image.Resampling.LANCZOS)
    arr = np.array(img_rgb, dtype=float)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Luminance (ITU-R BT.709)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mean_lum = float(np.mean(luminance))
    contrast_std = float(np.std(luminance))

    # Color temperature metric: Warm (R > B) vs Cool (B > R)
    warm_ratio = float(np.mean(r) / (np.mean(b) + 1e-5))

    if warm_ratio > 1.25:
        temp_tag = "Warm (Fiery / Energetic / Romantic)"
    elif warm_ratio < 0.85:
        temp_tag = "Cool (Cold / Futuristic / Atmospheric)"
    else:
        temp_tag = "Neutral / Balanced"

    if mean_lum < 70:
        lighting_tag = "Low-Key / Noir Shadows (Thriller, Horror, Crime)"
    elif mean_lum > 150:
        lighting_tag = "High-Key / Bright Lighting (Comedy, Adventure)"
    else:
        lighting_tag = "Mid-Tone Cinematic Contrast (Drama, Action)"

    return {
        "mean_luminance": round(mean_lum, 1),
        "contrast_index": round(contrast_std, 1),
        "warm_ratio": round(warm_ratio, 2),
        "temperature_tag": temp_tag,
        "lighting_tag": lighting_tag
    }


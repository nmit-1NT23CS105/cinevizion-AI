import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ============================================================
# CINEVISION AI - EXPLAINABILITY & VISUALIZATION UTILITY
# ============================================================

def create_aesthetic_radar_chart(genres, probabilities, thresholds=None):
    """
    Renders an 8-axis Plotly Polar Radar Chart showing genre confidence footprint.
    """
    if not HAS_PLOTLY:
        return None

    # Close the polygon loop by appending the first element
    r_vals = list(probabilities) + [probabilities[0]]
    theta_vals = list(genres) + [genres[0]]

    fig = go.Figure()

    # Predicted Probabilities trace
    fig.add_trace(go.Scatterpolar(
        r=r_vals,
        theta=theta_vals,
        fill="toself",
        name="Confidence",
        line=dict(color="#3b82f6", width=2.5),
        fillcolor="rgba(59, 130, 246, 0.35)"
    ))

    # Optional Threshold trace
    if thresholds is not None:
        t_vals = [thresholds.get(g, 0.5) for g in genres]
        t_vals.append(t_vals[0])
        fig.add_trace(go.Scatterpolar(
            r=t_vals,
            theta=theta_vals,
            mode="lines",
            name="Decision Threshold",
            line=dict(color="#ef4444", width=1.5, dash="dash")
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                tickfont=dict(size=9, color="#9ca3af"),
                gridcolor="#374151"
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#e5e7eb", family="sans-serif"),
                gridcolor="#374151"
            ),
            bgcolor="#111827"
        ),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color="#d1d5db")
        ),
        height=360
    )

    return fig

def generate_gradcam_heatmap(keras_model, img_batch, class_idx, is_efficientnet=True):
    """
    Generates a Grad-CAM heatmap array for a specific class index.
    """
    try:
        if is_efficientnet:
            try:
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
                        if layer.name in ["poster_input", "efficientnetb0", "letterbox_poster_input"]:
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
            except Exception:
                pass

        # Fallback for custom CNN, ResNet-SE, or direct conv model
        with tf.GradientTape() as tape:
            x = img_batch
            conv_output = None
            for layer in keras_model.layers:
                x = layer(x)
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.ReLU)) and len(x.shape) == 4:
                    conv_output = x
            if conv_output is not None:
                tape.watch(conv_output)
            loss = x[:, class_idx]

        if conv_output is not None:
            grads = tape.gradient(loss, conv_output)
            if grads is not None:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0)
                max_val = tf.reduce_max(heatmap)
                if max_val > 0:
                    heatmap /= max_val
                return heatmap.numpy()
        return None

    except Exception:
        return None

def create_heatmap_overlay(pil_img, heatmap_np, alpha=0.4):
    """
    Overlays a Grad-CAM heatmap onto a PIL image.
    """
    w, h = pil_img.size
    heatmap_resized = cv2.resize(heatmap_np, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    img_np = np.array(pil_img.convert("RGB"))
    overlay = cv2.addWeighted(img_np, 1.0 - alpha, heatmap_color, alpha, 0)
    return heatmap_color, overlay


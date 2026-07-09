# Model loading, Grad-CAM generation, image preprocessing/quality checks,
# and severity/warning estimation.

import io
import json
import base64
import logging
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

from PIL import Image
import cv2
import tensorflow as tf

from config import MODELS_DIR, IMG_SIZE

logger = logging.getLogger(__name__)

# ── Load Models ───────────────────────────────────────────────────────────────
main_model = None
validator = None
grad_model = None
class_names = []

logger.info("Loading models...")
try:
    # Limit TensorFlow memory usage
    import os
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # Limit memory growth
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'

    main_model = tf.keras.models.load_model(
        str(MODELS_DIR / "best_model.keras"),
        compile=False  # saves memory
    )
    validator = tf.keras.models.load_model(
        str(MODELS_DIR / "paddy_validator.keras"),
        compile=False  # saves memory
    )
    with open(MODELS_DIR / "class_names.json") as f:
        class_names = json.load(f)

    # Initialize Grad-CAM model once
    last_conv_layer = None
    for layer in reversed(main_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is not None:
        grad_model = tf.keras.models.Model(
            inputs  = main_model.inputs,
            outputs = [main_model.get_layer(last_conv_layer).output, main_model.output]
        )
        logger.info("Grad-CAM model initialized successfully!")
    else:
        logger.warning("No convolutional layer found for Grad-CAM initialization.")

    logger.info("Models loaded successfully!")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise


# ── Image Quality Check ───────────────────────────────────────────────────────
def check_image_quality(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = img_array.mean()
    if blur_score < 50:
        return False, "Image is too blurry. Please retake with a steady hand."
    if mean_brightness < 30:
        return False, "Image is too dark. Please improve lighting."
    if mean_brightness > 220:
        return False, "Image is overexposed. Please reduce lighting."
    return True, "OK"


# ── Severity Estimation ───────────────────────────────────────────────────────
def get_severity(confidence, disease):
    if disease == "Healthy":
        return "None"
    if confidence >= 90:
        return "Severe"
    elif confidence >= 75:
        return "Moderate"
    else:
        return "Mild"


# ── Warning ────────────────────────────────────────────────────────────────────
def get_warning(disease, confidence):
    if disease == "Blast" and confidence < 92:
        return "Blast can resemble Brownspot. Please verify with an agricultural expert."
    if disease == "Tungro":
        return "Tungro spreads rapidly. Isolate affected area and contact expert immediately."
    return None


# ── Grad-CAM ───────────────────────────────────────────────────────────────────
def generate_gradcam(img_array, class_idx):
    try:
        if grad_model is None:
            return None

        img_tensor = tf.cast(np.expand_dims(img_array, axis=0), tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, class_idx]

        grads        = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap      = heatmap.numpy()

        heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # ── Return ONLY the superimposed heatmap ──
        original     = (img_array * 255).astype(np.uint8)
        superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

        buf = io.BytesIO()
        Image.fromarray(superimposed).save(buf, format='JPEG', quality=75)
        buf.seek(0)
        result = base64.b64encode(buf.read()).decode('utf-8')
        del conv_outputs, grads, pooled_grads, heatmap
        del heatmap_resized, heatmap_colored, superimposed
        import gc; gc.collect()
        return result

    except Exception as e:
        logger.error(f"Grad-CAM error: {e}")
        return None


# ── Preprocess ─────────────────────────────────────────────────────────────────
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array   = np.array(img_resized) / 255.0
    return img_array
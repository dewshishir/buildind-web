"""
Flask backend for segmentation model inference
Production-ready for Render deployment
"""

import os
import sys
import time
import base64
import io
import cv2
import torch
import numpy as np
import gdown
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# -------------------------------------------------
# Add project root to Python path
# -------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import SegModel

# -------------------------------------------------
# Initialize Flask app
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------
# Configuration
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    "model.ckpt"
)

IMG_SIZE = 512
model = None

# -------------------------------------------------
# Download Model from Google Drive
# -------------------------------------------------
def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")

        url = "https://drive.google.com/uc?id=1SIYgEU3UdMaxCN7G00sdv6PsBnuYfngH"
        gdown.download(url, MODEL_PATH, quiet=False)

        print("Model downloaded successfully.")
    else:
        print("Model already exists. Skipping download.")

# -------------------------------------------------
# Load Model
# -------------------------------------------------
def load_model():
    global model
    try:
        download_model()

        model = SegModel.load_from_checkpoint(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )

        model.to(DEVICE)
        model.eval()

        print(f"Model loaded successfully on {DEVICE}")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
def preprocess_image(image_array):
    resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    resized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)

# -------------------------------------------------
# Inference
# -------------------------------------------------
def predict_mask(image_array):
    with torch.inference_mode():
        tensor = preprocess_image(image_array)
        pred = model(tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
    return pred.squeeze().cpu().numpy()

# -------------------------------------------------
# Base64 Helpers
# -------------------------------------------------
def base64_to_image(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def numpy_to_base64(array):
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)

    image = Image.fromarray(array)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": model is not None
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()

        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "Missing image data"}), 400

        image_array = base64_to_image(data["image"])
        prediction = predict_mask(image_array)

        pred_base64 = numpy_to_base64(prediction)
        confidence = float(np.mean(prediction))

        inference_time = time.time() - start_time

        return jsonify({
            "success": True,
            "prediction": f"data:image/png;base64,{pred_base64}",
            "confidence": confidence,
            "inference_time": round(inference_time, 3),
            "device": str(DEVICE)
        })

    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict-with-gt", methods=["POST"])
def predict_with_gt():
    try:
        start_time = time.time()

        data = request.json
        if not data or "image" not in data or "gt_mask" not in data:
            return jsonify({"error": "Missing image or gt_mask data"}), 400

        image_array = base64_to_image(data["image"])
        gt_array = base64_to_image(data["gt_mask"])

        gt_resized = cv2.resize(gt_array, (IMG_SIZE, IMG_SIZE))
        if len(gt_resized.shape) == 3:
            gt_resized = cv2.cvtColor(gt_resized, cv2.COLOR_BGR2GRAY)

        gt_normalized = gt_resized / 255.0
        prediction = predict_mask(image_array)

        intersection = np.logical_and(prediction > 0.5, gt_normalized > 0.5).sum()
        union = np.logical_or(prediction > 0.5, gt_normalized > 0.5).sum()

        iou = intersection / union if union > 0 else 0.0
        dice = 2 * intersection / (prediction.sum() + gt_normalized.sum() + 1e-8)

        pred_base64 = numpy_to_base64(prediction)
        inference_time = time.time() - start_time

        return jsonify({
            "success": True,
            "prediction": f"data:image/png;base64,{pred_base64}",
            "confidence": float(np.mean(prediction)),
            "iou": float(iou),
            "dice": float(dice),
            "inference_time": round(inference_time, 3),
            "device": str(DEVICE)
        })

    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# Load model when Gunicorn starts
# -------------------------------------------------
print("Initializing model...")
load_model()

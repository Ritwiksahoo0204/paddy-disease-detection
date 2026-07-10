# 🌾 Paddy Doctor — Paddy Leaf Disease Detection

A full-stack deep learning web application that detects paddy (rice) leaf diseases from a photo, explains the result with a visual heatmap, and provides multilingual treatment guidance — built as a final year college project.

**Live demo:** `https://paddy-disease-detection.vercel.app`
**Backend API:** `https://paddy-doctor-backend.onrender.com`

---

## Overview

Paddy Doctor lets a farmer or user upload a photo of a paddy leaf (via camera or gallery) and get back:
- The predicted disease (or "Healthy")
- A confidence score and severity level
- A Grad-CAM heatmap showing which part of the leaf influenced the prediction
- Symptoms, treatment steps, and prevention tips — in **English, Hindi, or Bengali**
- A scan history and an admin analytics dashboard

The project evolved from model training/experimentation in Google Colab into a complete, deployed full-stack application with a FastAPI backend and a React frontend.

---

## Features

- 📷 Camera capture or gallery upload for leaf images
- 🧠 Two-stage ML pipeline: a binary "is this a paddy leaf?" validator, then a 5-class disease classifier
- 🔥 Grad-CAM visual explanations for every prediction
- 🌐 Multilingual treatment content (English / Hindi / Bengali)
- 📊 Per-device scan history (no login required)
- 🔐 Admin dashboard with usage statistics (auth-protected)
- ✅ Image quality checks (blur, brightness) before prediction
- ⚡ Rate limiting on the prediction endpoint

---

## Tech Stack

**Backend (production app)**
- FastAPI (Python) — REST API
- TensorFlow / Keras — model inference
- OpenCV, Pillow — image processing
- SQLite — scan history and stats storage
- slowapi — rate limiting
- Deployed on **Render**

**Frontend (production app)**
- React + Vite
- React Router
- Recharts — admin dashboard charts
- Deployed on **Vercel**

**Model Development**
- Google Colab, TensorFlow/Keras, tf.data pipeline
- MobileNetV2 (transfer learning)
- Streamlit / Plotly used during early experimentation and visualization in Colab (not part of the deployed app)

---

## Dataset

| Detail | Info |
|---|---|
| Source | Mendeley Rice Leaf Disease Dataset + Kaggle Paddy Doctor Dataset (Healthy class) |
| Total Images | 21,974 |
| Total Classes | 5 |
| Input Size | 224 × 224 × 3 |
| Train / Validation Split | 80% / 20% |
| Training Batches | 550 |
| Validation Batches | 138 |
| Batch Shape | (32, 224, 224, 3) |

### Class Distribution

| Class | Images |
|---|---|
| Bacterial Blight | 4347 |
| Blast | 4778 |
| Brown Spot | 6469 |
| Healthy | 2952 |
| Tungro | 3428 |
| **Total** | **21,974** |

### Preprocessing & Augmentation

- Pixel values normalized from 0–255 to 0–1
- Class weights computed to handle mild class imbalance:

| Class | Weight |
|---|---|
| Bacterial Blight | 1.0110 |
| Blast | 0.9198 |
| Brown Spot | 0.6794 |
| Healthy | 1.4887 |
| Tungro | 1.2820 |

| Augmentation | Value |
|---|---|
| Rotation | 20° |
| Width shift | 10% |
| Height shift | 10% |
| Zoom | 15% |
| Horizontal flip | Yes |
| Vertical flip | Yes |
| Brightness range | 0.8 – 1.2 |

---

## Model Architecture & Training

### Disease Classifier
- **Base:** MobileNetV2, pretrained on ImageNet
- **Custom head:** GAP → BatchNorm → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Softmax(5)
- **Data pipeline:** `tf.data` (replacing `ImageDataGenerator` for GPU-parallel loading)

| Training Phase | Epochs | Learning Rate | Layers Trained |
|---|---|---|---|
| Phase 1 — Frozen base | 15 | 1e-4 | Custom head only |
| Phase 2 — Fine-tuning | 16 → 39 | 1e-5 | Last 30 MobileNetV2 layers |

| Metric | Value |
|---|---|
| Validation Accuracy | **94.26%** |
| Validation Loss | 0.1624 |
| Model Size | ~25 MB |

### Paddy Validator (Binary Gate)

A separate binary classifier filters out non-paddy images before they reach the disease classifier.

- **Architecture:** MobileNetV2 (frozen) → GAP → Dropout(0.3) → Dense(1, sigmoid)
- **Decision threshold:** trained/evaluated at 0.6 (probability ≥ 0.6 → treated as a paddy leaf); the deployed app raises this to **0.85** as a mitigation against false positives on out-of-distribution images (see Known Limitations)

| Validator Dataset | Images |
|---|---|
| Paddy (200 × 5 classes) | 1000 |
| Not Paddy | 1000 |
| **Total** | **2000** |

**Flow:**
```
User uploads image
        ↓
Validator checks probability
        ↓ prob ≥ threshold        ↓ prob < threshold
Main classifier              Reject image
predicts disease             show error message
```

---

## Project Structure

```
paddy-disease-detection/
├── backend/
│   ├── main.py              # App setup, CORS, middleware, router registration
│   ├── config.py            # Paths, constants, rate limiter
│   ├── database.py          # SQLite init/migrations
│   ├── auth.py               # Admin authentication (HTTPS-enforced)
│   ├── ml_utils.py           # Model loading, Grad-CAM, preprocessing
│   ├── treatments_data.py    # Disease info in English/Hindi/Bengali
│   ├── routes/
│   │   ├── predict.py        # /health, /predict
│   │   ├── history.py        # /feedback, /prediction/{id}, /history
│   │   └── dashboard.py      # /dashboard/stats
│   ├── models/                # Trained .keras models (classifier + validator)
│   ├── requirements.txt
│   ├── Procfile
│   └── runtime.txt
└── frontend/
    ├── src/
    │   ├── pages/             # Home, Camera, Analyzing, Result, Treatment, History, Dashboard
    │   ├── components/        # Header, BottomNav, DisclaimerModal
    │   └── utils/session.js   # Per-device anonymous session ID
    └── vite.config.js
```

---

## Getting Started (Local Development)

### Backend
```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt
py main.py
```
Runs on `http://localhost:8000`. Create a `.env` file with:
```
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<your password>
FRONTEND_URL=<your deployed frontend URL, if any>
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Runs on `http://localhost:5173`. Create a `.env` file with:
```
VITE_API_URL=http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Upload an image, get a prediction |
| GET | `/history` | Get scan history for the current device |
| GET | `/prediction/{id}` | Fetch a specific past prediction |
| POST | `/feedback/{id}` | Submit correct/incorrect feedback |
| GET | `/dashboard/stats` | Admin-only usage statistics |

---

## Known Limitations

- **Validator false positives:** the binary "is this a paddy leaf?" validator can occasionally misclassify out-of-distribution images (e.g. human faces) as paddy leaves, since its 2,000-image training set's negative examples did not include such categories. Mitigated in the deployed app by raising the decision threshold from 0.6 to 0.85; a more robust fix would involve retraining with a broader, more diverse set of negative examples.
- **Framing sensitivity:** the classifier performs best on close-up single-leaf photos. Wide-angle shots with multiple leaves in frame can bias predictions toward "Healthy" if the diseased leaf occupies a small portion of the image.

---

## Deployment

- **Backend:** Render (see `Procfile`, `runtime.txt`, `requirements.txt`)
- **Frontend:** Vercel (see `vercel.json`)

Both require the environment variables listed above to be set on the respective hosting platform for full functionality (admin login, CORS).
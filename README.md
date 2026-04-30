# 🌾 Paddy Disease Detection
### AI-Powered Rice Leaf Disease Classification Web App

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HuggingFace%20Spaces-FFD21E?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/rs60204/paddy-disease-detection)
[![GitHub](https://img.shields.io/badge/GitHub-Ritwiksahoo0204-181717?style=for-the-badge&logo=github)](https://github.com/Ritwiksahoo0204/paddy-disease-detection)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)

---

## 🔗 Live Demo

> **[👉 Click here to open the app](https://huggingface.co/spaces/rs60204/paddy-disease-detection)**

Upload a paddy leaf image and get an instant AI diagnosis — no installation required.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Disease Classes](#-disease-classes)
- [Project Structure](#-project-structure)
- [Pipeline Phases](#-pipeline-phases)
  - [Phase 1 — Dataset Setup](#phase-1--dataset-setup)
  - [Phase 2 — Preprocessing](#phase-2--preprocessing)
  - [Phase 3 — Model Building](#phase-3--model-building)
  - [Phase 4 — Evaluation](#phase-4--evaluation)
  - [Phase 5b — Validator Training](#phase-5b--validator-training)
  - [Phase 5 — Streamlit App](#phase-5--streamlit-app)
- [App Features](#-app-features)
- [Tech Stack](#-tech-stack)
- [Model Architecture](#-model-architecture)
- [Deployment](#-deployment)
- [Local Setup](#-local-setup)
- [Environment Variables](#-environment-variables)

---

## 🌟 Overview

Paddy Disease Detection is an end-to-end deep learning project that classifies rice leaf diseases from images using a fine-tuned **MobileNetV2** model. The system includes a **two-stage inference pipeline** — first validating whether the uploaded image is actually a paddy leaf, then classifying the disease — wrapped in a polished **Streamlit** web application with user authentication, activity history, and an admin panel.

---

## 🦠 Disease Classes

The model classifies paddy leaves into **5 categories**:

| # | Class | Description |
|---|-------|-------------|
| 1 | 🔴 **Bacterial Blight** | Water-soaked lesions turning yellow-brown along leaf margins |
| 2 | 🟠 **Blast** | Diamond-shaped grey/white lesions with brown borders |
| 3 | 🟣 **Brown Spot** | Circular brown spots scattered across the leaf |
| 4 | 🟢 **Healthy** | No disease — normal green paddy leaf |
| 5 | 🔵 **Tungro** | Yellow-orange discolouration caused by viral infection |

---

## 📁 Project Structure

```
paddy-disease-detection/
│
├── Phase1_Dataset_Setup.ipynb       # Dataset exploration & visualization
├── Phase2_Preprocessing.ipynb       # Image augmentation & data pipeline
├── Phase3_Model_Building.ipynb      # MobileNetV2 training & fine-tuning
├── Phase4_Evaluation.ipynb          # Model evaluation & metrics
├── Phase5b_Validator_Training.ipynb # Binary paddy/not-paddy validator
├── Phase5_Streamlit.ipynb           # Full app code + HuggingFace deployment
│
├── Phase1_outputs/
│   ├── class_distribution.png
│   └── sample_images.png
│
├── Phase2_outputs/
│   └── augmented_samples.png
│
├── Phase3_outputs/
│   ├── best_model.keras             # Main 5-class disease classifier
│   ├── paddy_validator.keras        # Binary paddy/not-paddy validator
│   └── class_names.json
│
├── Phase4_outputs/
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   └── sample_predictions.png
│
├── app.py                           # Streamlit application
├── database.py                      # SQLite DB — users, activity, visits
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔄 Pipeline Phases

### Phase 1 — Dataset Setup

**Notebook:** `Phase1_Dataset_Setup.ipynb`

- Mounts Google Drive and loads the rice disease dataset
- Dataset located at `Project_2k26/rice_disease_dataset/`
- Counts images per class and flags any class with fewer than 100 images
- Generates a **class distribution bar chart** and **sample image grid** (one per class)
- Saves all outputs to `Phase1_outputs/` in Drive and pushes to GitHub

**Config:**
```
Classes    : Bacterialblight, Blast, Brownspot, Healthy, Tungro
Image size : 224 × 224
Batch size : 32
Seed       : 42
```

---

### Phase 2 — Preprocessing

**Notebook:** `Phase2_Preprocessing.ipynb`

- Builds **training and validation data generators** using `ImageDataGenerator`
- Applies **data augmentation** to training images only:
  - Random rotation (±20°)
  - Width & height shift (10%)
  - Zoom (15%)
  - Horizontal & vertical flip
  - Brightness variation (0.8–1.2×)
  - Fill mode: nearest
- Validation generator applies **only rescaling** (no augmentation)
- Computes **class weights** using sklearn's `compute_class_weight('balanced')` to handle class imbalance
- Saves augmented sample visualization to `Phase2_outputs/`

**Train/Val Split:** 80% training / 20% validation

---

### Phase 3 — Model Building

**Notebook:** `Phase3_Model_Building.ipynb`

Uses **Transfer Learning** with MobileNetV2 in a **two-phase training strategy**:

**Phase 3.1 — Frozen Base (15 epochs)**
- Loads MobileNetV2 pretrained on ImageNet (`include_top=False`)
- Freezes all base model layers
- Trains only the custom classification head

**Custom Classification Head:**
```
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.4)
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(5, activation='softmax')
```

**Phase 3.2 — Fine-Tuning (30 epochs)**
- Unfreezes the **top 30 layers** of the base model
- Recompiles with a much lower learning rate (`1e-5`)
- Continues training on the full dataset

**Callbacks used:**
- `ModelCheckpoint` — saves best model by `val_accuracy`
- `EarlyStopping` — patience of 7 epochs
- `ReduceLROnPlateau` — reduces LR by factor 0.3 if val_loss stagnates (patience 3)

**Output:** `best_model.keras` + `class_names.json`

---

### Phase 4 — Evaluation

**Notebook:** `Phase4_Evaluation.ipynb`

Loads the best saved model and runs full evaluation on the validation set:

- **Overall accuracy** computed using `sklearn.metrics.accuracy_score`
- **Classification report** — per-class precision, recall, F1-score
- **Confusion matrix** — heatmap visualized with seaborn
- **Per-class accuracy bar chart** — coloured per class
- **Sample predictions grid** (2×5) — green title = correct, red = wrong

All plots saved to `Phase4_outputs/` and pushed to GitHub.

---

### Phase 5b — Validator Training

**Notebook:** `Phase5b_Validator_Training.ipynb`

Trains a **separate binary classifier** to detect whether an uploaded image is actually a paddy leaf or not — preventing the main model from predicting disease on completely unrelated images (e.g. a photo of a car).

**Dataset construction:**
- **Paddy class:** 200 images sampled from each of the 5 disease classes → 1,000 total
- **Not-paddy class:** 1,000 images from a separate ZIP of non-paddy images (rice/paddy related folders excluded)

**Model:** MobileNetV2 (frozen) → GlobalAveragePooling2D → Dropout(0.3) → Dense(1, sigmoid)

**Threshold:** `prob ≥ 0.6` → classified as paddy leaf

**Output:** `paddy_validator.keras`

---

### Phase 5 — Streamlit App

**Notebook:** `Phase5_Streamlit.ipynb`

Writes and deploys the full web application. Includes `app.py`, `database.py`, `requirements.txt`, `Dockerfile`, and pushes everything to HuggingFace Spaces via `git push`.

---

## ✨ App Features

### 🔍 Disease Detection Page
- Upload a paddy leaf image (JPG/PNG)
- **Two-stage inference:**
  1. `paddy_validator.keras` checks if the image is a paddy leaf
  2. `best_model.keras` predicts the disease class
- Shows predicted disease, confidence score, and severity label
- Displays an **interactive confidence bar chart** (Plotly) for all 5 classes
- Logs each prediction to the user's activity history

### 📖 My History Page
- Shows the last 50 predictions made by the logged-in user
- Columns: image name, predicted class, confidence, severity, timestamp

### 👑 Admin Panel (admin only)
- View all registered users with creation date and last login
- View full activity log (last 200 records across all users)
- **Force-reset any user's password** (bypasses old password requirement)
- **Download the raw SQLite database** as a `.db` file

### 🔐 Authentication System
- **Login**, **Sign Up**, and **Password Reset** — all on tabbed UI
- Passwords hashed with `bcrypt`
- Username validation: alphanumeric + underscore, 3–20 characters
- Minimum password length: 6 characters
- Admin credentials loaded from **environment variables** (not hardcoded)

### 🎨 Theme Toggle
- Switch between **🌑 Dark mode** and **☀️ Light mode** from the sidebar
- Custom CSS applied across sidebar, tabs, upload box, and Plotly charts

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow / Keras |
| Base Model | MobileNetV2 (ImageNet weights) |
| Web Framework | Streamlit 1.40.0 |
| Charts | Plotly |
| Database | SQLite3 |
| Auth | bcrypt |
| Containerisation | Docker (python:3.11-slim) |
| Deployment | HuggingFace Spaces |
| Storage | HuggingFace Storage Buckets (`/data`) |
| Training Environment | Google Colab (GPU) |

---

## 🧠 Model Architecture

```
Input: 224 × 224 × 3 RGB image
         ↓
MobileNetV2 (pretrained, top 30 layers fine-tuned)
         ↓
GlobalAveragePooling2D
         ↓
BatchNormalization
         ↓
Dense(256) → ReLU → Dropout(0.4)
         ↓
Dense(128) → ReLU → Dropout(0.3)
         ↓
Dense(5) → Softmax
         ↓
Output: [Bacterialblight, Blast, Brownspot, Healthy, Tungro]
```

**Validator (binary):**
```
Input: 224 × 224 × 3
         ↓
MobileNetV2 (frozen)
         ↓
GlobalAveragePooling2D → Dropout(0.3)
         ↓
Dense(1) → Sigmoid
         ↓
Output: paddy (≥0.6) / not-paddy (<0.6)
```

---

## 🚀 Deployment

The app is deployed on **HuggingFace Spaces** using a custom Docker container.

**Dockerfile summary:**
```dockerfile
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get install -y libgl1 libglib2.0-0
COPY requirements.txt . && pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Persistent Storage:** HuggingFace Storage Bucket mounted at `/data` → SQLite database saved at `/data/paddy_app.db` (survives Space restarts).

---

## 💻 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Ritwiksahoo0204/paddy-disease-detection.git
cd paddy-disease-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export ADMIN_USERNAME=your_admin_username
export ADMIN_PASSWORD=your_strong_password

# 4. Run the app
streamlit run app.py
```

**requirements.txt:**
```
streamlit==1.40.0
tensorflow
numpy
pillow
plotly
bcrypt
```

---

## 🔑 Environment Variables

Set these in **HuggingFace Space Settings → Variables and Secrets** before running:

| Variable | Type | Description |
|----------|------|-------------|
| `ADMIN_USERNAME` | Variable | Username for the admin account |
| `ADMIN_PASSWORD` | Secret 🔒 | Password for the admin account |

> ⚠️ Always add `ADMIN_PASSWORD` as a **Secret** (encrypted), never as a plain Variable.

---

## 👤 Author

**Ritwik Sahoo**
📧 ritwiksahoo2004@gmail.com
🔗 [GitHub](https://github.com/Ritwiksahoo0204) | [HuggingFace](https://huggingface.co/rs60204)

---

> *Built as part of Project 2K26 — an end-to-end ML project covering data setup, preprocessing, model training, evaluation, and production deployment.*

# Paddy Disease Detection 🌾

## 🔴 Live Demo
> [Click here to open the app](https://huggingface.co/spaces/rs60204/paddy-disease-detection)

## ⚠️ Important
This app works with **paddy (rice) leaves only**.
All other images are automatically rejected.

## 4-Step Prediction Pipeline
| Step | Check | Action if Failed |
|------|-------|-----------------|
| 1 | Image quality | Rejects blurry / dark images |
| 2 | Paddy validator | Rejects non-paddy images |
| 3 | Disease prediction | Classifies disease |
| 4 | Confidence threshold | Rejects if below 70% |

## Disease Classes & Accuracy
| Class | Accuracy |
|-------|----------|
| Bacterialblight | 99.37% |
| Blast | 98.61% |
| Brownspot | 100.0% |
| Healthy | 100.0% |
| Tungro | 100.0% |
| **Overall** | **99.58%** |

## Model Details
| Detail | Info |
|--------|------|
| Architecture | MobileNetV2 (Transfer Learning) |
| Pretrained on | ImageNet |
| Training Images | 7,220 |
| Overall Accuracy | 99.58% |
| Validator | Binary Paddy Classifier |

## Project Phases
| Phase | Task | Status |
|-------|------|--------|
| Phase 1 | Dataset Setup | ✅ Done |
| Phase 2 | Preprocessing | ✅ Done |
| Phase 3 | Model Building | ✅ Done |
| Phase 4 | Evaluation — 99.58% | ✅ Done |
| Phase 5 | Streamlit Web App | ✅ Done |
| Phase 6 | Deployment (HF Spaces) | ✅ Done |

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
Python • TensorFlow • Keras • MobileNetV2 • Streamlit • Plotly • OpenCV • Docker • Hugging Face Spaces

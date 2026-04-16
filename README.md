# Paddy Disease Detection 🌾

## Live Demo
> Deployment link will be added after Phase 6

## ⚠️ Important
This app is designed for **paddy (rice) leaves only**.
- Other plant leaves will be **automatically rejected**
- Irrelevant images (faces, objects) will be **rejected**
- Low confidence predictions are **rejected**

## Disease Classes
| Class | Per-Class Accuracy |
|-------|--------------------|
| Bacterialblight | 99.37% |
| Blast | 98.61% |
| Brownspot | 100.0% |
| Healthy | 100.0% |
| Tungro | 100.0% |
| **Overall** | **99.58%** |

## How It Works (4-Step Pipeline)
1. Image quality check — rejects blurry/dark images
2. Paddy validator — rejects non-paddy leaf images
3. Disease prediction — MobileNetV2 classifies disease
4. Confidence threshold — rejects low confidence results

## Project Phases
| Phase | Task | Status |
|-------|------|--------|
| Phase 1 | Dataset Setup | ✅ Done |
| Phase 2 | Preprocessing | ✅ Done |
| Phase 3 | Model Building | ✅ Done |
| Phase 4 | Evaluation — 99.58% | ✅ Done |
| Phase 5 | Streamlit Web App | ✅ Done |
| Phase 6 | Deployment | ⏳ Pending |

## Model Details
| Detail | Info |
|--------|------|
| Architecture | MobileNetV2 (Transfer Learning) |
| Framework | TensorFlow / Keras |
| Overall Accuracy | 99.58% |
| Input Size | 224 x 224 x 3 |
| Min Confidence | 70% |

## How to Run Locally


## Tech Stack
- Python 3 / TensorFlow / Keras
- MobileNetV2 Transfer Learning
- Binary Paddy Validator
- Streamlit / Plotly / OpenCV
- Google Colab

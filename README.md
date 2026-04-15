# Paddy Disease Detection

## Project Overview
A deep learning project to detect paddy leaf diseases using MobileNetV2 transfer learning.
Built using TensorFlow, Keras, and Streamlit.

## Disease Classes
| Class | Images | Status |
|-------|--------|--------|
| Bacterialblight | 1584 | OK |
| Blast | 1440 | OK |
| Brownspot | 1400 | OK |
| Healthy | 1488 | OK |
| Tungro | 1308 | OK |
| TOTAL | 7220 | OK |

## Phase 1 - Dataset Setup (Completed)
- Total images : 7220 across 5 classes
- Well balanced dataset, no class imbalance issues

## Phase 2 - Preprocessing (Completed)
- Normalized pixel values, applied data augmentation
- 80% train / 20% validation split
- Computed class weights for imbalance handling

## Phase 3 - Model Building (Completed)
- Architecture  : MobileNetV2 (Transfer Learning)
- Pretrained on : ImageNet
- Custom head   : GAP → BatchNorm → Dense(256) → Dropout → Dense(128) → Dropout → Softmax
- Training      : 2-phase (frozen base → fine-tuning top 30 layers)
- Model size    : 25.03 MB

## Phase 4 - Evaluation (Completed)
- Overall Accuracy : 99.58%

### Per Class Accuracy
| Class | Accuracy |
|-------|----------|
| Bacterialblight | 99.37% |
| Blast | 98.61% |
| Brownspot | 100.00% |
| Healthy | 100.00% |
| Tungro | 100.00% |

## Upcoming Phases
- Phase 5 : Streamlit Web App
- Phase 6 : Deployment on Streamlit Cloud

## Dataset Details
| Detail | Info |
|--------|------|
| Source | Mendeley + Kaggle |
| Total Images | 7220 |
| Input Size | 224 x 224 x 3 |
| Train Split | 80% |
| Validation Split | 20% |

## Tech Stack
- Python 3
- TensorFlow / Keras
- MobileNetV2
- Streamlit
- Google Colab
- Plotly

# Paddy Disease Detection

## Project Overview
A deep learning project to detect paddy leaf diseases using MobileNetV2 transfer learning.
Built using TensorFlow, Keras, and Streamlit.

## Disease Classes
| Class | Images | Status |
|-------|--------|--------|
| Bacterialblight | 4347 | OK |
| Blast | 4778 | OK |
| Brownspot | 6469 | OK |
| Healthy | 2952 | OK |
| Tungro | 3428 | OK |
| TOTAL | 21974 | OK |

## Phase 1 - Dataset Setup (Completed)
- Collected dataset from Mendeley Rice Leaf Disease Dataset
- Added Healthy class from Kaggle Paddy Doctor dataset
- Total images : 22,000+
- Total classes : 5
- Dataset is well balanced across all classes
- No class imbalance issues detected

### Phase 1 Outputs
| File | Description |
|------|-------------|
| class_distribution.png | Bar chart showing image count per class |
| sample_images.png | One sample image from each disease class |

## Upcoming Phases
- Phase 2 : Data Preprocessing and Augmentation
- Phase 3 : Model Building (MobileNetV2)
- Phase 4 : Model Evaluation
- Phase 5 : React App
- Phase 6 : Deployment

## Dataset Details
| Detail | Info |
|--------|------|
| Source | Mendeley + Kaggle |
| Total Images | 21,000+ |
| Image Format | JPG / PNG |
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

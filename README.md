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
- Total images : 21,974
- Total classes : 5
- Dataset is well balanced across all classes
- No class imbalance issues detected

## Phase 2 - Preprocessing (Completed)
- Normalized pixel values from 0-255 to 0-1
- Applied data augmentation techniques
- Split dataset 80% train / 20% validation
- Training batches   : 550
- Validation batches : 138
- Batch image shape  : (32, 224, 224, 3)
- Computed class weights for imbalance handling

### Class Weights
| Class | Weight |
|-------|--------|
| Bacterialblight | 1.0110 |
| Blast | 0.9198 |
| Brownspot | 0.6794 |
| Healthy | 1.4887 |
| Tungro | 1.2820 |

### Augmentation Techniques Used
| Technique | Value |
|-----------|-------|
| Rotation | 20 degrees |
| Width shift | 10% |
| Height shift | 10% |
| Zoom | 15% |
| Horizontal flip | Yes |
| Vertical flip | Yes |
| Brightness range | 0.8 to 1.2 |

## Phase 3 - Model Building (Completed)
- Architecture  : MobileNetV2 (Transfer Learning)
- Pretrained on : ImageNet
- Custom head   : GAP → BatchNorm → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Softmax(5)
- Data pipeline : tf.data (replaced ImageDataGenerator for GPU-parallel loading)

### Training Strategy
| Phase | Epochs | LR | Layers Trained |
|-------|--------|----|----------------|
| Phase 1 — Frozen | 15 | 1e-4 | Custom head only |
| Phase 2 — Fine-tune | 16 → 39 | 1e-5 | Last 30 MobileNetV2 layers |

### Final Results
| Metric | Value |
|--------|-------|
| Validation Accuracy | **94.26%** |
| Validation Loss | 0.1624 |

## Phase 3B - Validator Training (Completed)
- Binary classifier to reject non-paddy images
- Architecture : MobileNetV2 (frozen) → GAP → Dropout(0.3) → Dense(1, sigmoid)
- Threshold    : 0.6 (prob >= 0.6 → paddy, prob < 0.6 → not paddy)

### Validator Dataset
| Class | Images |
|-------|--------|
| Paddy | 1000 (200 × 5 classes) |
| Not Paddy | 1000 (from zip) |
| Total | 2000 |

### How Validator Works in App
```
User uploads image
        ↓
Validator checks probability
        ↓ prob >= 0.6          ↓ prob < 0.6
Main Model               Reject image
predicts disease         show error message
```

### Phase 3B - Output Files
| File | Description |
|------|-------------|
| paddy_validator.keras | Binary validator model |
| Phase3b_Validator_Training.ipynb | Training notebook |

## Upcoming Phases
- Phase 4 : Model Evaluation
- Phase 5 : React App
- Phase 6 : Deployment

## Dataset Details
| Detail | Info |
|--------|------|
| Source | Mendeley + Kaggle |
| Total Images | 21,974 |
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

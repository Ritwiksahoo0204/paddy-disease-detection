# Paddy Disease Detection 🌾

## Project Overview
A deep learning web app that detects paddy leaf diseases using MobileNetV2 transfer learning.
Built using TensorFlow, Keras, and Streamlit.

## Live Demo
> Deploy link will be added after Phase 6

## Disease Classes
| Class | Accuracy |
|-------|----------|
| Bacterialblight | 99.37% |
| Blast | 98.61% |
| Brownspot | 100.0% |
| Healthy | 100.0% |
| Tungro | 100.0% |
| **Overall** | **99.58%** |

## Phase 1 - Dataset Setup (Completed)
- Total images : 7220 across 5 classes
- Well balanced dataset, no class imbalance issues

## Phase 2 - Preprocessing (Completed)
- Normalized pixel values, applied data augmentation
- 80% train / 20% validation split

## Phase 3 - Model Building (Completed)
- Architecture  : MobileNetV2 (Transfer Learning)
- Pretrained on : ImageNet
- Model size    : 25.03 MB
- Training      : 2-phase frozen base + fine-tuning

## Phase 4 - Evaluation (Completed)
- Overall Accuracy : 99.58%
- Only 6 misclassifications out of 1442 samples

## Phase 5 - Streamlit Web App (Completed)
- Upload paddy leaf image
- Get instant disease prediction
- View confidence scores per class
- Read disease description, symptoms and treatment
- Interactive Plotly confidence chart

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
paddy-disease-detection/
├── app.py
├── best_model.keras
├── class_names.json
├── requirements.txt
├── Phase1_Dataset_Setup.ipynb
├── Phase2_Preprocessing.ipynb
├── Phase3_Model_Building.ipynb
├── Phase4_Evaluation.ipynb
├── Phase5_Streamlit.ipynb
├── Phase1_outputs/
├── Phase2_outputs/
├── Phase3_outputs/
└── Phase4_outputs/

## Tech Stack
- Python 3
- TensorFlow / Keras
- MobileNetV2
- Streamlit
- Plotly
- Google Colab

# ASL Sign Language Recognition

Realtime American Sign Language (ASL) Recognition System using Computer Vision and Machine Learning.

## Project Overview

This project implements a realtime ASL gesture recognition pipeline capable of detecting hand gestures from webcam input and predicting corresponding ASL alphabet signs.

The system was developed as part of:

- CSR311 — Artificial Intelligence and Machine Learning II
- CSR322 — Natural Language Processing
- Lovely Professional University (LPU)

The project focuses on:

- Computer Vision
- Gesture Recognition
- Machine Learning
- Realtime Webcam Inference
- Human-Computer Interaction

---

# Features

- Realtime webcam-based ASL recognition
- Hand-region detection using OpenCV
- Gesture preprocessing pipeline
- Machine Learning based gesture classification
- Lightweight realtime inference
- Dataset preprocessing and feature extraction
- Train/test ML pipeline
- Interactive live prediction UI

---

# Project Pipeline

```text
Webcam Input
      ↓
Image Preprocessing
      ↓
Hand Region Detection
      ↓
Feature Extraction
      ↓
Machine Learning Classification
      ↓
ASL Gesture Prediction
```

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Computer vision and webcam processing |
| NumPy | Numerical computation |
| Pandas | Dataset processing |
| Scikit-learn | Machine learning model training |
| Joblib | Model serialization |
| VS Code | Development environment |
| Arch Linux | Development platform |

---

# Dataset

Dataset used:

ASL Alphabet Dataset (Kaggle)

- 87,000+ gesture images
- 29 gesture classes
- Includes:
  - A–Z alphabets
  - space
  - delete
  - nothing

Dataset Link:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

---

# Project Structure

```text
ASL_Sign_Language_Recognition/
│
├── asl_data/
│   ├── asl_alphabet_train/
│   └── asl_alphabet_test/
│
├── outputs/
├── venv/
│
├── extract_landmarks.py
├── train_model.py
├── detect.py
│
├── asl_landmarks.csv
├── asl_model.pkl
│
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repository-link>
cd ASL_Sign_Language_Recognition
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas scikit-learn joblib opencv-python
```

---

# Running the Project

## Step 1 — Extract Features

```bash
python extract_landmarks.py
```

This creates:

```text
asl_landmarks.csv
```

---

## Step 2 — Train Model

```bash
python train_model.py
```

This creates:

```text
asl_model.pkl
```

---

## Step 3 — Run Realtime Detection

```bash
python detect.py
```

---

# Model Approach

The deployed realtime system uses:

- OpenCV contour-based hand detection
- ROI extraction
- Image preprocessing
- Feature vector generation
- Random Forest classifier for ASL prediction

The lightweight pipeline was selected to ensure stable realtime inference on local hardware.

---

# Preprocessing Techniques

- Grayscale conversion
- Gaussian Blur
- Thresholding
- Contour detection
- Image resizing
- Feature flattening

---

# Machine Learning Pipeline

1. Dataset preprocessing
2. Feature extraction
3. Train/test split
4. Random Forest model training
5. Realtime webcam inference
6. Gesture prediction

---

# Results

- Realtime gesture prediction achieved successfully
- Stable webcam-based inference
- Lightweight CPU-friendly pipeline
- Fast training compared to deep CNN approaches

---

# Challenges Faced

- Realtime inference optimization
- Environment compatibility on Arch Linux
- Webcam preprocessing stability
- Gesture/background separation
- Managing latency during live prediction

---

# Future Improvements

- MediaPipe hand landmark integration
- CNN-based deep learning classifier
- Temporal gesture recognition
- Sentence generation using NLP
- Mobile deployment
- Voice output integration
- Transformer-based gesture recognition

---

# Screenshots

Add your screenshots here:

- Dataset samples
- Training output
- Live webcam prediction
- Detection interface

---

# Authors

## Thubalami Nkomanzana
- B.Tech CSE (AI & Data Engineering)
- Lovely Professional University

## Rohan Ajith Shankar Pilli
- Lovely Professional University

---

# Academic Context

This project was developed for:

- CSR311 — Artificial Intelligence and Machine Learning
# 🤟 ASL Sign Language Recognition with Gesture-to-Text Translation

> A full pipeline that takes American Sign Language hand gesture images, recognizes the letter being signed using a custom-trained CNN, and translates sequences of letters into grammatically coherent sentences using NLP techniques.

Built as part of the **LPU CSE AI/ML Integrated Project 2025–26** (Project P07).

---

## 📌 Project Overview

This project is split into two tightly connected parts:

| Part | Owner | Focus |
|------|-------|-------|
| **ML / Deep Learning** | Thubalami Nkomazana | CNN training, Grad-CAM, accuracy evaluation |
| **NLP** | Rohan Ajith Shankar Pilli | Sequence-to-word translation, grammar checking, sentence coherence |

The CNN from Part 1 feeds directly into the NLP pipeline of Part 2 — letter predictions become word completions, which become grammatically checked sentences.

---

## 🧠 Part 1 — Deep Learning (CNN for ASL Classification)

### What it does
Trains a custom Convolutional Neural Network from scratch on 87,000 ASL hand sign images to classify 29 classes — the full alphabet (A–Z) plus `del`, `nothing`, and `space`.

### Architecture

nput  (3 × 64 × 64)
→ Block 1: Conv(32)  → BN → ReLU → MaxPool   (32 × 32 × 32)
→ Block 2: Conv(64)  → BN → ReLU → MaxPool   (64 × 16 × 16)
→ Block 3: Conv(128) → BN → ReLU → MaxPool   (128 × 8 × 8)
→ Block 4: Conv(256) → BN → ReLU → MaxPool   (256 × 4 × 4)
→ Global Average Pooling                      (256,)
→ FC(512) → Dropout(0.5) → FC(29)

**~1.35M trainable parameters**

### Training Setup

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 with ReduceLROnPlateau |
| Epochs | 20 |
| Batch Size | 64 |
| Image Size | 64 × 64 |
| Loss | CrossEntropy + Label Smoothing (0.1) |
| Regularization | Dropout(0.5) + Weight Decay(1e-4) |

### Data Augmentation
- Random horizontal flip (p=0.5)
- Random rotation ±15°
- Color jitter (brightness, contrast, saturation)
- ImageNet normalization

### Results

| Metric | Value |
|--------|-------|
| Top-1 Validation Accuracy | ~94% |
| Most confused pairs | M↔N, S↔E, D↔F |

### Visualizations

The notebook produces 5 output visualizations:

| File | Description |
|------|-------------|
| `training_curves.png` | Loss and accuracy over 20 epochs |
| `per_class_accuracy.png` | Per-class bar chart (color coded) |
| `confusion_matrix.png` | 29×29 normalized confusion matrix |
| `misclassified.png` | Grid of wrong predictions with true/predicted labels |
| `gradcam.png` | Grad-CAM heatmaps showing which hand regions activate each class |

---

## 💬 Part 2 — NLP (Gesture-to-Sentence Translation)

### What it does
Takes a sequence of CNN letter predictions and builds meaningful words and sentences from them using language modeling and grammar checking.

### Pipeline
CNN letter predictions
→ n-gram language model (word completion from partial letter sequences)
→ LanguageTool / spaCy grammar checking
→ fine-tuned DistilBERT (coherence classification)
→ final sentence output

### Evaluation Metrics
- **Character Error Rate (CER)** — how accurate individual letter predictions are
- **Word Error Rate (WER)** — how accurate the final word/sentence reconstruction is

---


## 🚀 Getting Started

### Prerequisites
- Google Colab with **T4 GPU** (recommended) or a local machine with CUDA
- A Kaggle account with API access

### Run Part 1 (DL)
1. Open `ASL Recognition.ipynb` in Google Colab
2. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
3. Run all cells top to bottom — the Kaggle token is already embedded
4. Training takes ~60–90 minutes. The best model is auto-saved as `asl_cnn.pth`

---

## 📊 Dataset

**ASL Alphabet** by grassknoted on Kaggle  
🔗 https://www.kaggle.com/datasets/grassknoted/asl-alphabet

- 87,000 training images
- 29 classes (A–Z + del, nothing, space)
- 200 × 200 px RGB images
- ~3,000 images per class

---

## 🔬 Key Techniques

**Deep Learning**
- Custom CNN built in PyTorch (no pretrained backbone)
- Grad-CAM for model interpretability
- ReduceLROnPlateau scheduling
- Label smoothing for better generalization

**NLP**
- n-gram language model for word completion
- LanguageTool API / spaCy for grammar checking
- DistilBERT fine-tuned for sentence coherence detection
- CER and WER evaluation metrics

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)

- **PyTorch** — model building and training
- **torchvision** — data loading and transforms
- **OpenCV** — image processing for Grad-CAM
- **HuggingFace Transformers** — DistilBERT fine-tuning
- **spaCy / LanguageTool** — grammar checking
- **scikit-learn** — evaluation metrics and confusion matrix
- **Matplotlib / Seaborn** — visualizations

---

## 👥 Team

| Name | Role | LinkedIn |
|------|------|---------|
| Thubalami Nkomazana | ML / Deep Learning | [thubalami-nkomazana15](https://www.linkedin.com/in/thubalami-nkomazana15) |
| Rohan Ajith Shankar Pilli | NLP Pipeline | [rohan-ajith-shankar](https://www.linkedin.com/in/rohan-ajith-shankar) |

*LPU CSE AI/ML — Integrated Project 2025–26*

---

## 📄 License

This project is for academic purposes as part of the LPU CSE curriculum.

# OCR-Devanagari
# OCR for Handwritten Devanagari Script

A deep learning-based Optical Character Recognition (OCR) system for **handwritten Devanagari text**, built using a **CRNN architecture (CNN + BiLSTM + CTC Loss)**.

---

## Overview

This project presents an end-to-end OCR pipeline specifically designed for handwritten Devanagari script (Hindi, Sanskrit, Marathi, Nepali, etc.).

Unlike generic OCR systems, this model is:

- Script-specific
- Lightweight (~4.75M parameters)
- Trained from scratch (no pretrained backbone)
- Open and accessible

---

## Results

| Metric | Value |
|--------|------|
| CER (Greedy) | **11.1%** |
| CER (Beam Search) | **~9%** |
| Word Error Rate (WER) | **38%** |
| Exact Word Match | **62%** |

Best performance was achieved at **Epoch 23**.

---

## Motivation

- 600M+ people use Devanagari script
- Large volumes of handwritten documents remain undigitized
- Existing OCR systems:
  - Are not optimized for Devanagari
  - Perform poorly on cursive handwriting
- Cloud APIs are expensive and not accessible to everyone

This project provides a **free, accurate, and specialized OCR solution**.

```text
Dataset → Preprocessing → Vocabulary → Model → Evaluation
```
## 🏗️ Pipeline

```text
Dataset → Preprocessing → Vocabulary → Model → Evaluation
```

### Workflow

1. Dataset Acquisition using the **IIIT-HW-Dev** dataset with ~95K word images
2. Preprocessing with grayscale conversion, resizing, and normalization
3. Data augmentation using blur, rotation, perspective, and sharpness changes
4. Vocabulary building with ~97 characters plus the CTC blank token
5. Model training using **CNN + BiLSTM + CTC**
6. Evaluation using **CER / WER** and decoding methods

---

## 📂 Dataset

**IIIT-HW-Dev Dataset**

- 95,000+ handwritten word images
- Train: 69,853
- Validation: 12,708
- Test: 12,869

### Annotation Format

```text
<image_path> <label>
HindiSeg/train/1/234/7.jpg दवाओं
```

- UTF-8 encoded Devanagari labels
- Grayscale word images with variable width

---

## 🧠 Model Architecture (CRNN)

```text
Input Image (1×32×128)
        ↓
CNN Backbone
        ↓
Sequence Conversion
        ↓
BiLSTM (2 layers)
        ↓
Fully Connected Layer
        ↓
CTC Loss
```

### Details

- CNN: 5 convolutional blocks with 64 → 512 channels
- BiLSTM:
  - 2 layers
  - 256 hidden units per direction
- Output sequence length: 32
- Vocabulary size: ~97
- Total parameters: ~4.75M

---

## 🔗 CTC Loss

CTC (Connectionist Temporal Classification) allows training without character-level alignment.

- Supports blank tokens and repeated characters
- No need for manual segmentation
- Useful for sequence prediction where input and output lengths differ

### Example

```text
Raw:      र र - - ा ा म म
Decoded:  राम
```

---

## 🧪 Preprocessing & Augmentation

### Preprocessing

- Convert RGB images to grayscale
- Resize to **32 × 128**
- Normalize to the range **[-1, 1]**

### Augmentation

- Gaussian blur
- Color jitter
- Random rotation (±4°)
- Random perspective transform
- Sharpness adjustment

---

## ⚡ Training Details

- GPU: NVIDIA RTX 3050 Ti / Quadro T2000
- Training time: less than 3 hours
- Epochs: 50

---

## 📈 Evaluation Metrics

- Character Error Rate (CER)
- Word Error Rate (WER)
- Beam search improves accuracy over greedy decoding

---

## 🔮 Future Work

- Transformer-based OCR to replace BiLSTM
- Lexicon-guided decoding
- Line-level OCR
- Synthetic data generation
- Mobile deployment using ONNX / TFLite

---

## 📁 Repository Structure

```text
├── data/
├── models/
├── training/
├── inference/
├── utils/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/devanagari-ocr.git
cd devanagari-ocr
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train

```bash
python train.py
```

### Inference

```bash
python inference.py --image path/to/image.jpg
```

---

## 👨‍💻 Authors

- Kumar Ayush
- Devesh Dhurandher

---

## ⭐ Key Insight

> A small, well-trained, script-specific model can outperform large generic OCR systems.

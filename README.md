# OCR-Devanagari
# 🧠 OCR for Handwritten Devanagari Script

A deep learning-based Optical Character Recognition (OCR) system for **handwritten Devanagari text**, built using a **CRNN architecture (CNN + BiLSTM + CTC Loss)**.

---

## 🚀 Overview

This project presents an end-to-end OCR pipeline specifically designed for handwritten Devanagari script (Hindi, Sanskrit, Marathi, Nepali, etc.).

Unlike generic OCR systems, this model is:

- Script-specific
- Lightweight (~4.75M parameters)
- Trained from scratch (no pretrained backbone)
- Open and accessible

---

## 📊 Results

| Metric | Value |
|--------|------|
| CER (Greedy) | **11.1%** |
| CER (Beam Search) | **~9%** |
| Word Error Rate (WER) | **38%** |
| Exact Word Match | **62%** |

Best performance was achieved at **Epoch 23**.

---

## 🎯 Motivation

- 600M+ people use Devanagari script
- Large volumes of handwritten documents remain undigitized
- Existing OCR systems:
  - Are not optimized for Devanagari
  - Perform poorly on cursive handwriting
- Cloud APIs are expensive and not accessible to everyone

This project provides a **free, accurate, and specialized OCR solution**.

```text
Dataset → Preprocessing → Vocabulary → Model → Evaluation

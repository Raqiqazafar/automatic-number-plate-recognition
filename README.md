# 🚗 Automatic Number Plate Recognition (ANPR)

An AI-based system for real-time vehicle license plate detection and text extraction using **YOLOv11** and **EasyOCR**.

---

## 📌 Project Overview

A complete ANPR pipeline — from dataset preparation to license plate text extraction — capable of handling real-world challenges like varied lighting, angles, and occlusion.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| YOLOv11 | License plate detection |
| EasyOCR | Text extraction from plates |
| Matplotlib | Visualization with bounding boxes |
| Google Colab | Training environment (GPU T4) |
| Google Drive | Dataset storage and management |

---

## ⚙️ Project Pipeline

```
Step 1: Mount Google Drive & Load Dataset
        ↓
Step 2: Dataset Splitting (Train / Val / Test)
        ↓
Step 3: YOLOv11 Model Training
        ↓
Step 4: Validation & Testing
        ↓
Step 5: EasyOCR Text Extraction
        ↓
Step 6: Matplotlib Visualization with Bounding Boxes
```

---

## 🔍 Key Features

- Real-time license plate **detection** using YOLOv11
- **Text extraction** from detected plates using EasyOCR
- Dataset split into **train/val/test** sets for proper model evaluation
- Handles real-world challenges — **lighting variations, occlusion, angles**
- Visualizations with **bounding boxes** using Matplotlib
- Trained on **Google Colab T4 GPU** for fast processing

---

## 📊 Sample Output

```
Detected Number Plate Text: ECB:025
```

---

## 🧩 Key Learnings

- Hands-on experience with **YOLOv11** for object detection
- Proper **dataset splitting** strategy for deep learning models
- Integration of **EasyOCR** for optical character recognition
- **Bounding box visualization** with Matplotlib
- Handling real-world image processing challenges






> 🏫 Superior University Lahore — Digital Image Processing, Fall 2025
> 
> 🔗 LinkedIn: [Raqiqa Zafar](https://www.linkedin.com/in/raqiqa-zafar)

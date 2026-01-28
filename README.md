# Maltese Traffic Signs Detection Project
## Advanced Computer Vision (ARI3129)

**Project Title:** Automatic Detection & Attribute Classification of Maltese Traffic Signs

**Team Members:** Luca Bugeja, Liam Debono, Matthias Ellul, Remi Heijmans

**Deadline:** 31st January 2026

---

## 📋 Project Overview

This project focuses on using computer vision techniques to detect and classify traffic signs in Malta, along with their attributes (viewing angle, mounting type, sign condition, and sign shape).

### Project Repository Structure

```
Final-CV-Project-Submission/
├── 1_data_visualisation.ipynb               # Dataset statistics
├── 2a_[architecture_name]__[student].ipynb  # Object detection models (one per student - four total)
├── 2b_[attribute]_[student].ipynb           # Attribute classification models (one per student - four total)
├── 2c_results_comparison.ipynb              # Compare all models
├── ARI3129 Group 6 Documentation.pdf        # Group Documentation
├── ARI3129 Group 6 GenAI Journal.pdf        # Group Generative AI Journal
├── Maltese Traffic Signs Dataset/           # Link to the dataset folder
│   ├── Individuals/                         # Images & Annotations of each member
│   │   ├── images_Liam_Debono.zip
│   │   ├── images_Luca_Bugeja.zip
│   │   ├── images_Matthias_Ellul.zip
│   │   ├── images_Remi_Heijmans.zip
│   │   ├── input_Liam_Debono.json
│   │   ├── input_Luca_Bugeja.json
│   │   ├── input_Matthias_Ellul.json
│   │   ├── input_Remi_Heijmans.json
│   ├── COCO Exports/                        # Various annotations in COCO format
│   │   ├── COCO.json     
│   │   ├── COCO_mounting.json
│   │   ├── COCO_sign_condition.json
│   │   ├── COCO_sign_shape.json
│   │   ├── COCO_sign_type.json
│   │   ├── COCO_viewing_angle.json
│   ├── merged_input.json                    # All annotations               
│   ├── merged_images.zip                    # All captured images
└── README.md                                # This file
```

---

## 📊 Task 1: Dataset Preparation

### Captured Traffic Signs

The dataset includes the following Maltese traffic signs:
- **Stop Sign**
- **No Entry Sign** (One Way)
- **Pedestrian Crossing** (Zebra Crossing)
- **Roundabout Ahead**
- **No Through Road** (T-Sign)
- **Blind-Spot Mirrors** (Convex Mirrors)

### Sign Attributes Annotated

Each sign is labeled with:
1. **Sign Type:** The specific sign category
2. **Viewing Angle:** Front, Back, or Side
3. **Mounting Type:** Wall-mounted or Pole-mounted
4. **Sign Condition:** Good, Weathered, or Heavily Damaged
5. **Sign Shape:** Circular, Square, Triangular, Octagonal, or Damaged

### Dataset Statistics

- **Total Images:** 620
- **Total Annotations:** 654
- **Images per Team Member:** 50+ unique physical signs
- **Split:** 70% Train / 15% Validation / 15% Test

### Dataset Visualization

```bash
jupyter notebook 1_data_visualisation.ipynb
```

This notebook generates:
- Distribution tables for all sign types and attributes
- Sample annotated images
- Dataset balance analysis

---

## 🎯 Task 2: Object Detection

### 2a: Traffic Sign Detection Models

Each team member trains a different object detection model, tasked with detecting and classifying traffic signs:

#### Matthias Ellul: YOLOv8
```bash
jupyter notebook 2a_yolov8_matthias.ipynb
```
- **Architecture:** YOLOv8n (Nano)
- **Achieved mAP50:** 72.4%

#### Luca Bugeja: YOLOv11
```bash
jupyter notebook 2a_yolov11_LucaBugeja.ipynb
```
- **Architecture:** YOLOv11n
- **Achieved mAP50:** 85%

#### Liam Debono: Faster R-CNN
```bash
jupyter notebook 2a_fasterrcnn_Liam.ipynb
```
- **Architecture:** Faster R-CNN (ResNet50 backbone)
- **Achieved mAP50:** 30.9%

#### Remi Heijmans: Faster R-CNN
```bash
jupyter notebook 2a_faster_rcnn_remi.ipynb
```
- **Architecture:** RetinaNet (ResNet50 backbone)
- **Achieved mAP50:** 59.4%


---

## 🏷️ Task 2b: Attribute Classification

Each team member trains a classifier for one specific attribute:

### Matthias Ellul: Sign Shape Classification
```bash
jupyter notebook 2b_sign_shape_matthias.ipynb
```
- **Architecture:** TF / Keras
- **Classes:** Circular, Square, Triangular, Octagonal, Damaged
- **Achieved Accuracy:** 86.9%

### Luca Bugeja: Mounting Type Classification
```bash
jupyter notebook 2b_MountingType_LucaBugeja.ipynb
```
- **Architecture:** YOLOv11
- **Classes:** Wall-mounted, Pole-mounted
- **Achieved Accuracy:** 85%

### Liam Debono: Sign Condition Classification
```bash
jupyter notebook 2b_sign_condition_Liam.ipynb
```
- **Architecture:** PyTorch
- **Classes:** Good, Weathered, Heavily Damaged
- **Achieved Accuracy:** 79.6%

### Remi Heijmans: Viewing Angle Classification
```bash
jupyter notebook 2b_viewing_angle_remi.ipynb
```
- **Architecture:** YOLOv8
- **Classes:** Front, Back, Side
- **Achieved Accuracy:** 91.4%
---

## 📈 Task 2c: Results Comparison

```bash
jupyter notebook 2c_results_comparison.ipynb
```

This notebook:
- Compares all object detection models
- Compares all attribute classification models
- Generates comprehensive visualizations
- Ranks models by performance

---

## 📝 Documentation

### Main Documentation (20 pages max)

Located in: `ARI3129 Group6 Documentation.pdf`

**Contents:**
1. Introduction
2. Background on the Techniques Used
3. Data Preparation
4. Implementation of the Object Detectors
5. Evaluation of the Object Detectors
6. References and List of Resources Used

### Generative AI Journal (10 pages max)

Located in: `ARI3129 Group6 GenAI Journal.pdf`

**Contents:**
1. Introduction
2. Ethical Considerations
3. Methodology
4. Prompts and Responses
5. Improvements, Errors, and Contributions
6. Individual Reflection
7. References and Resources Used

---

## 🔬 Evaluation Metrics

### Object Detection
- **mAP@0.5** (Mean Average Precision at IoU=0.5)
- **mAP@0.5:0.95** (Mean Average Precision at varying IoU thresholds from 0.5 to 0.95)
- **Precision** - True Positives / (True Positives + False Positives)
- **Recall** - True Positives / (True Positives + False Negatives)
- **F1-Score** - 2 × (Precision × Recall) / (Precision + Recall)

### Attribute Classification
- **Confusion Matrix** - Detailed classification breakdown
- **Per-class Precision, Recall, F1-Score**


---

## 📚 Resources

### Official Documentation
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Label Studio](https://labelstud.io/guide/)
- [PyTorch](https://pytorch.org/docs/)
- [OpenCV](https://docs.opencv.org/)

### Tutorials
- [YOLO Object Detection Tutorial](https://www.ultralytics.com/yolo)
- [Custom Dataset Training](https://docs.ultralytics.com/modes/train/)
- [Model Evaluation Metrics](https://docs.ultralytics.com/modes/val/)

### Papers
- YOLOv8: [Ultralytics YOLO](https://arxiv.org/abs/2305.09972)
- Faster R-CNN: [Towards Real-Time Object Detection](https://arxiv.org/abs/1506.01497)
- RetinaNet: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

---

## 👥 Team Contribution Summary

| Student | Detector | Attribute |
|---------|----------|-----------|
| Matthias Ellul | YOLOv8 | TF / Keras for Sign Shape |
| Luca Bugeja | YOLOv11 | YOLOv11 for Mounting Type |
| Liam Debono | Faster R-CNN | PyTorch for Sign Condition |
| Remi Heijmans | RetinaNet | YOLOv8 for Viewing Angle |

---

## 📄 License & Data Privacy

- All captured images comply with GDPR regulations
- Personal information (faces, license plates) has been removed/blurred
- Dataset is for educational purposes only
- External images are properly cited

---

## ✅ Submission Checklist

- ✅ All 50+ images per team member captured
- ✅ Complete annotations in Label Studio
- ✅ COCO format export completed
- ✅ All notebooks executed successfully
- ✅ Visualizations generated
- ✅ Models trained and evaluated
- ✅ Documentation written (max 20 pages)
- ✅ GenAI Journal completed (max 10 pages)
- ✅ GitHub repository organized
- ✅ README.md updated
- ✅ All team member contributions documented
---


# Brain Tumor Recognition and Classification

## Project Overview

This project focuses on the automated recognition and classification of brain tumors using deep learning techniques. It addresses the real-world challenge of limited labeled medical data by applying a **SimCLR-based self-supervised learning approach** combined with **10-Shot-Learning** and **lightweight CNN backbones**.

Given the high stakes of early and accurate tumor detection in clinical environments, this system aims to:
- Detect whether an MRI image contains a tumor.
- Classify the tumor type (e.g., Glioma, Meningioma, Pituitary) if present.
- Provide visual explanations using Grad-CAM to highlight the regions of interest contributing to the model’s prediction.

Despite the constraint of having only **100-150 MRI images**, the model leverages strong data augmentation and contrastive representation learning (SimCLR) to learn discriminative features without relying on pretrained weights — in compliance with project constraints.

## Dataset Description

This project uses the [Brain Tumor MRI Dataset by Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), sourced from Kaggle. The dataset contains T1-weighted contrast-enhanced MRI scans of human brains, categorized into four classes:

| Class      | Description                                                                               |
|------------|-------------------------------------------------------------------------------------------|
| Glioma     | A tumor that originates in the glial cells of the brain.                                  |
| Meningioma | A typically benign tumor arising from the meninges (the membranes surrounding the brain). |
| Pituitary  | Tumors located in the pituitary gland, often affecting hormonal balance.                  |
| No Tumor   | MRI images without visible tumors (healthy brain scans).                                  | 

```
Brain-Tumor-MRI-Dataset/
├── glioma_tumor/
├── meningioma_tumor/
├── pituitary_tumor/
└── no_tumor/
```

- Total Images: ~3,000
- Format: .jpg
- Image Size: Varies, typically grayscale or RGB
- Preprocessing: Resize → Augmentation → Normalization

### Note
Due to project constraints, **only a subset of the dataset was used** (approximately 150–200 images total). This limited-data setup simulates a **few-shot learning environment**, suitable for self-supervised learning approaches like **SimCLR**.

## Result and Comparison

# 🩺 Kidney Stone Detection from Ultrasound Images using Deep Learning

Welcome to the **Kidney Stone Detection** project! This repository provides an end-to-end deep learning pipeline to classify kidney ultrasound images into two categories: **Stone** and **Normal**. The models are trained and evaluated using modern deep learning techniques and are deployable via a lightweight Streamlit application.

---

## 📁 Dataset Overview

- **Total Images:** 9,416 ultrasound images  
- **Normal:** 4,414 images  
- **Stone:** 5,002 images  
- **Image Size:** All resized to `224x224`  
- **Sources:** Collected from various clinical devices including Samsung RS85, HS60, RS80A, and HS70A  
- **License:** CC BY 4.0  

---

## 🎯 Objective

**Binary Classification Task**

- 🔍 **Goal:** Automatically detect presence of kidney stones from ultrasound images.
- ⚙️ **Methods Used:**
  - Transfer Learning with **MobileNetV2**
  - A custom-built **CNN model**
  - Data Augmentation via `ImageDataGenerator`
  - Class imbalance handled with **class weights**
  - Multiple evaluation metrics for robust validation

---

## 🧠 Model Architectures

### 🔹 MobileNetV2 (Pre-trained on ImageNet)
- `include_top=False`, followed by `GlobalAveragePooling2D`
- Dense → Dropout → Dense (sigmoid)
- Optimizer: **Adam**
- Loss: **Binary Crossentropy**
- EarlyStopping + ReduceLROnPlateau + ModelCheckpoint

### 🔹 Custom CNN Model
- Sequential layers of:
  - Conv2D → BatchNorm → MaxPooling → Dropout
  - Final Dense layer with sigmoid activation
- Designed to be lightweight yet performant

---

## 📊 Results & Comparison

| Model        | Train Accuracy | Validation Accuracy | Test Accuracy |
|--------------|----------------|---------------------|---------------|
| MobileNetV2  | 98.30%         | 99.52%              | 99.26%        |
| Custom CNN   | 99.52%         | 99.36%              | 99.36%        |

### 📌 Observations:
- **MobileNetV2** generalizes slightly better on the validation set.
- **Custom CNN** shows exceptional performance on the test set with fewer parameters.


---

## 🧪 Evaluation Metrics

* ✅ Accuracy
* 📉 Loss Curves
* 📈 ROC Curve, AUC
* 📊 Confusion Matrix
* 🔬 Precision, Recall, F1-Score

---

## 🗂 Project Structure

```
├── app/                  # Streamlit app
├── datasets/             # Original and processed images
├── models/               # Saved model files (.h5)
├── notebooks/            # Training and evaluation notebooks
├── requirements/         # requirements.txt for dependencies
└── README.md             # Project documentation

```


## 📚 Dependencies

* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn
* Streamlit

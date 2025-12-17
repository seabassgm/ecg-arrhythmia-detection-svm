# ECG Arrhythmia Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Librosa-Audio_Analysis-orange)
![Library](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-green)

> **📄 [Click here to read the full Project Report (PDF)](./docs/Gomez_Vitalone.pdf)**

## 📌 Project Overview
This project implements a supervised learning pipeline to classify cardiac anomalies using the **MIT-BIH Arrhythmia Database**. The primary challenge addressed was the severe class imbalance between normal heartbeats and arrhythmia conditions.

We engineered a custom pipeline involving signal processing, algorithmic data augmentation, and machine learning classification to detect 5 distinct arrhythmia classes:
* **N:** Normal
* **S:** Supraventricular premature beat
* **V:** Premature ventricular contraction
* **F:** Fusion of ventricular and normal beat
* **Q:** Unclassifiable beat

## 🔧 Methodology

### 1. Feature Extraction
We moved beyond simple raw signal processing by extracting a **30-point feature vector** focused on rhythmic properties:
* **Zero-Crossing Rate (ZCR):** To analyze oscillation frequency.
* **Spectral Flux:** To measure the change in spectrum magnitude between frames.
* **Statistical Moments:** Mean and Standard Deviation of the signal and derived features.

### 2. Custom Data Augmentation
To solve the dataset imbalance (where Normal beats vastly outnumbered others), we developed custom augmentation algorithms in Python:
* **Time Stretching:** Resampling signals to vary duration while preserving pitch.
* **Amplification:** Random gain staging to simulate signal strength variability.

### 3. Classification Model
We benchmarked several models, with the **Support Vector Machine (SVM)** utilizing an **RBF Kernel** ($C=1000$) achieving the robust performance.

## 📊 Key Results
* **Test Accuracy:** ~75.4% (Optimized SVM).
* **Minority Class Detection:** The augmentation strategy significantly improved the **Recall** and **F1-Scores** for minority classes (F and Q), reducing critical False Negatives compared to the baseline.
* **Generalization:** The RBF kernel successfully modeled the non-linear decision boundaries required to distinguish complex arrhythmia morphologies.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn librosa matplotlib scipy

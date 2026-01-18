# 🏥 Hospital OPD Early Warning System (3-Day Ahead Forecast)

An end-to-end Machine Learning system that predicts **hospital OPD / Emergency Department overload 3 days in advance** using only historical operational data.

This project is designed as a **realistic early warning system** for hospital administrators to anticipate congestion and take preventive actions (staffing, resource allocation, scheduling, etc.).

---

## Problem Statement

Hospital OPDs and Emergency Departments often face **sudden overload**, leading to:

- Long waiting times
- Resource exhaustion
- Reduced quality of care

The goal of this project is:

> To predict whether the hospital will be **overloaded 3 days in the future**, using only past and present data.

This avoids data leakage and makes the system **deployable in real-world settings**.

---

## Key Highlights

- Works on **143,000+ patient records**
- Uses **time-based train/test split**
- Predicts **3 days ahead**
- Full data cleaning + aggregation pipeline
- Rolling window features (3, 7, 14 days)
- Handles class imbalance using weighted Random Forest
- Evaluated using:
  - ROC-AUC
  - PR-AUC
  - Confusion Matrix
  - Classification Report
- Interpretable via **feature importance**
- Exports daily risk scores to CSV
_______________________________________________________________________________________________________________________________________________________________________
## System Pipeline
Raw Patient Data
        ↓
Daily Aggregation
        ↓
Rolling Feature Engineering
        ↓
Overload Definition (Top 10% stress days)
        ↓
3-Day Future Label Creation
        ↓
Time-based Train/Test Split
        ↓
ML Model Training (Random Forest)
        ↓
Evaluation + Risk Score Export
---

## Repository Structure

Hospital-OPD-Early-Warning-System/
│
├── main.py
├── data_loader.py
├── feature_engineering.py
├── labeling.py
├── model.py
├── evaluation.py
├── requirements.txt
└── daily_3day_ahead_risk.csv


---

## Features Used

- Daily arrivals
- Critical case ratio
- Average length of stay
- Lab, Instrument, Action, Imaging workload
- 7-day rolling averages of key signals

---

## Model Details

- **RandomForestClassifier**
- Class-weighted to handle imbalance
- Controlled depth to reduce overfitting
- Explainable using feature importance

---

## Typical Results

- ROC-AUC ≈ **0.81**
- PR-AUC ≈ **0.51**
- Accuracy ≈ **0.76**

These are **realistic results** for a real forecasting problem without data leakage.

---

## Suggested Visualizations

- Confusion Matrix
- Feature Importance Bar Chart
- Risk Score over Time
- Pipeline Diagram

---



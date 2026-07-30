# Breast Cancer Diagnosis Prediction

A machine learning project predicting whether a breast tumor is malignant or benign
based on clinical measurements, using the Wisconsin Breast Cancer dataset.

## Overview

- **Dataset:** 569 patient samples, 30 clinical measurements per tumor (radius, texture, smoothness, concavity, symmetry, etc.)
- **Model:** Logistic Regression (scikit-learn)
- **Result:** 97.4% test accuracy

## Motivation

This project was built to explore how computational tools can support clinically
meaningful decisions — a direction relevant to biomedical engineering. Rather than
just training a model, the goal was to understand *which* measurements actually
drive a diagnosis, and whether that lines up with real clinical knowledge.

## Method

1. Loaded the dataset (built into scikit-learn, originally from the UCI Machine
   Learning Repository)
2. Explored feature distributions and correlations
3. Split data 80/20 into training and test sets
4. Standardized features (important since measurements are on very different scales)
5. Trained a logistic regression classifier
6. Evaluated with accuracy, precision, and recall
7. Analyzed model coefficients to identify the most predictive features

## Results

| Metric | Malignant | Benign |
|---|---|---|
| Precision | 0.98 | 0.97 |
| Recall | 0.95 | 0.99 |
| F1-score | 0.96 | 0.98 |

**Overall accuracy: 97.4%**

### Top predictive features
1. Worst texture
2. Radius error
3. Worst symmetry
4. Mean concave points
5. Worst concavity

These results align with established clinical understanding: irregular cell shape
and inconsistent measurements across a tumor are hallmarks of malignancy.

## Visuals

- `diagnosis_distribution.png` — class balance in the dataset
- `correlation_heatmap.png` — how features relate to each other and to diagnosis
- `top_features.png` — which features most influenced the model's predictions

## How to run

```bash
pip install pandas scikit-learn matplotlib seaborn
python breast_cancer_analysis.py
```

## Data source

Wisconsin Breast Cancer Diagnostic dataset, available via
`sklearn.datasets.load_breast_cancer()` (originally from the UCI Machine Learning
Repository).

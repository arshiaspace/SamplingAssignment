# Sampling Assignment

## Objective
The objective of this assignment is to understand the importance of **sampling techniques** in handling **imbalanced datasets** and to analyze how different sampling strategies affect the performance of various machine learning models.

---

## Problem Statement
A highly imbalanced credit card dataset is provided. Such imbalance can significantly affect model performance.  
The task is to balance the dataset, apply different sampling techniques, train multiple machine learning models, and analyze how sampling impacts accuracy.

---

## Dataset
- **File:** `Creditcard_data.csv`
- **Target Column:** `Class`
  - `0` → Non-Fraud  
  - `1` → Fraud  

The dataset is balanced using **Random Under Sampling**.

---

## Sampling Techniques Used
1. **Sampling1:** Simple Random Sampling  
2. **Sampling2:** Systematic Sampling  
3. **Sampling3:** Stratified Sampling  
4. **Sampling4:** Bootstrap Sampling  
5. **Sampling5:** Cross-Validation Sampling  

---

## Machine Learning Models Used
- **M1:** Logistic Regression  
- **M2:** Decision Tree  
- **M3:** Random Forest  
- **M4:** Support Vector Machine (SVM)  
- **M5:** K-Nearest Neighbors (KNN)  

---

## Results

### Accuracy Comparison Table

| Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|------|----------|----------|----------|----------|----------|
| M1 | 83.33 | 77.78 | 50.00 | 62.50 | 50.00 |
| M2 | 16.67 | 77.78 | 83.33 | 25.00 | 75.00 |
| M3 | 16.67 | 77.78 | 66.67 | 37.50 | 50.00 |
| M4 | 33.33 | 44.44 | 33.33 | 50.00 | 50.00 |
| M5 | 50.00 | 66.67 | 50.00 | 50.00 | 50.00 |

---

## Best Model for Each Sampling Technique

- **Sampling1:** Best Model → **M1 (83.33%)**
- **Sampling2:** Best Model → **M1, M2, and M3 (77.78%)**
- **Sampling3:** Best Model → **M2 (83.33%)**
- **Sampling4:** Best Model → **M1 (62.50%)**
- **Sampling5:** Best Model → **M2 (75.00%)**

---

## Observations
- **Sampling1 (Simple Random Sampling)** performs best with **Logistic Regression (M1)**.
- **Sampling2 (Systematic Sampling)** shows similar performance across **M1, M2, and M3**.
- **Sampling3 (Stratified Sampling)** gives the highest accuracy when combined with **Decision Tree (M2)**.
- **Sampling4 (Bootstrap Sampling)** performs moderately, with **M1** giving the best result.
- **Sampling5 (Cross-Validation Sampling)** performs best with **Decision Tree (M2)**.


---

## How to Run the Code

### Install Dependencies
```bash
pip install -r requirements.txt

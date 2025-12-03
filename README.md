
# Enhanced Hybrid Deep Learning Ensemble for Credit Card Fraud Detection

**Authors:** Thabiso Msimango (u25738497) & Lungisani Khanyile (u25743695)  
**Institution:** University of Pretoria, Department of Computer Science  
**Course:** COS801 — Deep Learning  

## 1. Overview

Credit card fraud detection is challenging due to extremely imbalanced data (0.17% fraud rate) and constantly evolving attack patterns. Traditional deep learning models often prioritise recall but generate too many false positives, overwhelming fraud analysts.

This project presents a Hybrid Deep Learning Ensemble that integrates:

- **CNN** — extracts spatial patterns  
- **LSTM** — captures sequential spending behaviours  
- **Transformer Encoder** — models long-range dependencies using attention  
- **XGBoost Meta-Learner** — improves classification using stacked predictions  

We further introduce:

- **Focal Loss** (to emphasise minority samples)  
- **Threshold Calibration** using **F0.5 optimisation** (to prioritise precision)  

The resulting model reduces false positives by ~89%, making it substantially more practical for real-world banking environments.

## 2. Repository Structure

This project includes two key notebooks, each with a specific purpose:

### A. Baseline Model (Reference Paper Reproduction)

**File:** `Enhanced_Hybrid_Deep_Learning_Ensemble_for_Credit_Card_Fraud_Detection.ipynb`

- Replicates the hybrid ensemble from the reference paper  
- Uses conventional Binary Cross-Entropy loss  

**Expected Outcomes:**

- Recall ≈ 87%  
- Precision ≈ 33%  
- False Positives ≈ 171  

### B. Optimised Model (Our Contribution)

**File:** `Project_optimizationAUC.ipynb`

- Implements Focal Loss  
- Applies threshold tuning to maximise the **F0.5 Score**  

**Expected Outcomes:**

- Precision ≈ 81%  
- False Positives ≈ 18  
- **89% reduction** compared to the baseline  

## 3. Dataset

The project uses the **European Credit Card Fraud Dataset** from the ULB Machine Learning Group.

- 284,807 transactions  
- 492 fraud cases (0.172%)  
- 28 PCA-based features (V1–V28), plus Time and Amount  

The dataset is **not included** in this repository.  
Download it from Kaggle (`creditcard.csv`) and upload it before running the notebooks.

## 4. Installation & Setup

### Install Required Packages
```bash
pip install tensorflow xgboost scikit-learn pandas numpy matplotlib seaborn imbalanced-learn
```

### Running the Baseline Notebook

Open:  
`Enhanced_Hybrid_Deep_Learning_Ensemble_for_Credit_Card_Fraud_Detection.ipynb`

1. Upload `creditcard.csv`  
2. Run all cells  

### Running the Optimised Notebook

Open:  
`Project_optimizationAUC.ipynb`

1. Upload `creditcard.csv`  
2. Run all cells  

## 5. Results

### Performance Comparison

| Model                   | Recall   | Precision | F1 Score | AUC   |
|------------------------|----------|-----------|----------|-------|
| Logistic Regression     | 91.84%   | 6.08%     | 0.114    | 0.972 |
| Random Forest           | 74.44%   | 96.05%    | 0.840    | 0.953 |
| Baseline Hybrid Ensemble | 86.73%   | 33.20%    | 0.480    | 0.952 |
| Optimised Hybrid Ensemble | 76.53%   | 80.65%    | 0.785    | 0.966 |

### False Positive Reduction

- **Baseline:** 171 false positives  
- **Optimised:** 18 false positives  

→ **89% reduction**

This makes the optimised model much more suitable for practical fraud investigation workflows.

## 6. Future Enhancements

- Model explainability (SHAP values)  
- Cost-sensitive loss functions reflecting financial impact  
- Real-time inference optimisation for deployment in live banking systems  

## 7. Acknowledgements

- ULB Machine Learning Group (Dataset)  
- Ileberi & Sun (2024) — foundational hybrid ensemble research  
- Vaswani et al. (2017) — Transformer architecture  

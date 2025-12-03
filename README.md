# Enhanced Hybrid Deep Learning Ensemble for Credit Card Fraud Detection

**Authors:** Thabiso Msimango (u25738497) & Lungisani Khanyile
(u25743695)\
**Institution:** University of Pretoria - Department of Computer
Science\
**Course:** COS801 - Deep Learning

## 📌 Overview

Credit card fraud presents a massive challenge due to extreme class
imbalance (0.17% fraud rate) and evolving attack patterns. While deep
learning models can detect complex anomalies, they often suffer from
high false positive rates, which can overwhelm fraud analysts.

This project implements a **Hybrid Deep Learning Ensemble** combining:

1.  **CNN:** For spatial/local pattern extraction.\
2.  **LSTM:** For temporal/sequential spending patterns.\
3.  **Transformer:** For long-range dependency modeling via attention
    mechanisms.\
4.  **XGBoost:** As a meta-learner for final classification.

We propose an **Optimized Hybrid Ensemble** that introduces
imbalance-aware focal loss and precision-oriented threshold calibration.
This approach reduces false positives by **\~89%** compared to the
baseline ensemble while maintaining robust fraud detection capabilities.

------------------------------------------------------------------------

## 📂 Project Structure

  -----------------------------------------------------------------------------------------------------------------------------------
  File Name                                                                        Description & Purpose
  -------------------------------------------------------------------------------- --------------------------------------------------
  `Enhanced_Hybrid_Deep_Learning_Ensemble_for_Credit_Card_Fraud_Detection.ipynb`   **Baseline Model (Paper Reproduction)** ---
                                                                                   reproduces results from Ileberi & Sun (2024). Uses
                                                                                   Binary Cross-Entropy loss. Outcome: High Recall
                                                                                   (\~87%), Low Precision (\~33%), False Positives =
                                                                                   171.

  `Project_optimizationAUC.ipynb`                                                  **Optimized Model (Our Contribution)** ---
                                                                                   implements Focal Loss and Threshold Calibration.
                                                                                   Outcome: Precision (\~81%), False Positives = 18.
  -----------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 📊 Dataset

European Credit Card Fraud Detection Dataset (ULB Machine Learning
Group):

-   **Transactions:** 284,807\
-   **Fraud Cases:** 492 (0.172%)\
-   **Features:** PCA-transformed V1--V28 + Time + Amount

Dataset must be downloaded from Kaggle and uploaded manually.

------------------------------------------------------------------------

## 🚀 Installation & Usage

### Prerequisites

-   Python 3.8+
-   GPU recommended (Colab T4/A100)

### Install Dependencies

    pip install tensorflow xgboost scikit-learn pandas numpy matplotlib seaborn imbalanced-learn

### Running Experiments

#### Baseline Reproduction

1.  Open
    `Enhanced_Hybrid_Deep_Learning_Ensemble_for_Credit_Card_Fraud_Detection.ipynb`
2.  Upload dataset\
3.  Run all cells

Expected: High Recall (\~0.87), Low Precision (\~0.33)

#### Optimized Model

1.  Open `Project_optimizationAUC.ipynb`
2.  Upload dataset\
3.  Run all cells

Expected: Precision (\~0.81), False Positives (\~18)

------------------------------------------------------------------------

## 📈 Experimental Results

### Performance Metrics

  -------------------------------------------------------------------------------
  Model                Recall            Precision            F1          AUC
  --------------- ---------------- ---------------------- ----------- -----------
  Logistic             91.84%              6.08%             0.114       0.972
  Regression                                                          

  Random Forest        74.44%              96.05%            0.840       0.953

  Hybrid Ensemble      86.73%              33.20%            0.480       0.952
  (Baseline)                                                          

  **Hybrid           **76.53%**          **80.65%**        **0.785**   **0.966**
  Ensemble                                                            
  (Optimized)**                                                       
  -------------------------------------------------------------------------------

### Confusion Matrix Comparison

-   Baseline: **171 False Positives**\
-   Optimized: **18 False Positives** (89% reduction)

------------------------------------------------------------------------

## 🔮 Future Work

-   SHAP Explainability\
-   Cost-sensitive learning\
-   Real-time deployment optimizations

------------------------------------------------------------------------

## 📜 References & Acknowledgments

-   Ileberi & Sun (2024) --- Main reference\
-   Dataset: ULB Machine Learning Group\
-   Transformers: Vaswani et al. (2017)

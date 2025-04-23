# 🛒 MFG Data Science Challenge – Online Shoppers Purchasing Intention

This repository contains the code and analysis for the MFG Data Science Challenge (April 2025), in which we built predictive models to forecast whether an online user will complete a purchase.

## 📊 Project Description

We worked with the [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset) from the UCI repository. The task was framed as a binary classification problem (Purchase vs No Purchase), with a significant class imbalance (15.5% purchases).

## 🧪 Methodology

The workflow included:

- **EDA**: Distribution analysis, feature correlations, and behavioral insights
- **Preprocessing**: Categorical encoding, feature transformation, and stratified train/test split
- **Model Training**:
  - Logistic Regression (with and without SMOTE)
  - Random Forest
  - XGBoost
  - TabPFN (NeurIPS 2022 foundation model for tabular data)
- **Evaluation**:
  - Precision, Recall, F1-Score (focus on Recall for the positive class)
  - Feature importance comparison
  - Performance on full vs. reduced feature sets

## ⚙️ Models and Results

| Model            | Recall (Purchase) | F1 (Purchase) | Accuracy |
|------------------|------------------|----------------|----------|
| Logistic Regression      | 0.70              | 0.61           | 0.86     |
| Random Forest            | 0.76              | 0.66           | 0.88     |
| XGBoost                  | **0.84**          | 0.64           | 0.86     |
| TabPFN (Transformer)     | 0.72              | **0.67**       | **0.90** |

TabPFN achieved the highest F1-score, outperforming tuned traditional models without any hyperparameter tuning or class balancing.

## 💡 Key Insights

- PageValues is the most predictive feature across models.
- Feature selection reduced training time by ~40% without degrading performance.
- TabPFN showed exceptional robustness to data imbalance.
- XGBoost was the best model for minimizing false negatives.

## 🧠 Business Applications

- Real-time prediction of user purchase intent.
- Personalized offers and engagement based on prediction scores.
- Prioritization of hot leads for marketing and support.
- Enhanced A/B testing strategy via feature importance analysis.

## 📂 Files

- `MFG_report.pdf` – Full written report (in English).
- `notebooks/` – Jupyter notebooks for EDA, preprocessing, and modeling.
- `data/` – Cleaned or preprocessed datasets (if shared).
- `results/` – Plots, scores, and model comparisons.

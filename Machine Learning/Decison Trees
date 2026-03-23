Decision Tree Interview Preparation Guide

---

📚 Table of Contents

1. "Overview of Decision Trees" (#overview-of-decision-trees)
2. "Practical Applications" (#practical-applications)
3. "Data Preparation" (#data-preparation)
4. "Model Evaluation" (#model-evaluation)
5. "Common Interview Questions" (#common-interview-questions)

---

🌳 Overview of Decision Trees

Definition and Purpose

A Decision Tree is a supervised machine learning algorithm that uses a tree-like structure to make decisions.

- Splits data recursively based on feature values
- Internal nodes → decisions
- Branches → outcomes
- Leaf nodes → final prediction

🎯 Purpose

- Predict target variables using simple rules
- Highly interpretable (white-box model)
- Handles non-linear relationships
- Foundation for ensembles (Random Forest, XGBoost)

---

🔑 Key Concepts

Component| Description| Example
Root Node| First split of entire dataset| Income > 50K
Internal Node| Decision points| Age > 30
Branch| Outcome of split| Yes/No
Leaf Node| Final prediction| Approve Loan
Splitting| Dividing nodes based on criteria| Gini, Entropy
Pruning| Removing branches to reduce overfitting| Cost-complexity

---

🌲 Types of Decision Trees

1. Classification Trees

- Target: Categorical
- Criteria:
  - Gini Impurity
  - Entropy / Information Gain
- Output: Class label / probability

2. Regression Trees

- Target: Continuous
- Criteria:
  - MSE
  - MAE
- Output: Mean value of leaf

---

🏭 Practical Applications

Healthcare

- Disease prediction
- Patient readmission
- Treatment recommendation

Finance

- Credit scoring
- Fraud detection
- Customer segmentation

Marketing

- Churn prediction
- Recommendation systems
- Lead scoring

Manufacturing

- Predictive maintenance
- Quality control

---

💡 Why Decision Trees Work Well

- Interpretable
- Handles mixed data types
- Captures non-linearity
- Built-in feature importance
- Minimal preprocessing

---

🧹 Data Preparation

1. Data Cleaning

# Missing values
# Numerical → mean / median
# Categorical → mode / "Unknown"

# Remove duplicates
# Validate ranges

# Outliers
# Trees are robust, but extreme values may affect splits

---

2. Feature Engineering

- Binning (age groups)
- Interaction features
- Encoding:
  - Label Encoding (ordinal)
  - One-Hot Encoding (nominal)

---

3. Data Splitting

Train: 70–80%
Validation: 10–15%
Test: 10–20%

- Stratified sampling (classification)
- Time-based split (time series)

---

📌 Feature Selection

Methods:

- Filter → Correlation, Chi-square
- Wrapper → RFE
- Embedded → Tree importance, LASSO

---

📊 Model Evaluation

Classification Metrics

Metric| Formula| Use Case
Accuracy| (TP+TN)/Total| Balanced data
Precision| TP/(TP+FP)| FP costly
Recall| TP/(TP+FN)| FN costly
F1| Harmonic mean| Imbalanced
ROC-AUC| Curve area| Model comparison
Log Loss| Probabilities| Probabilistic models

---

Regression Metrics

Metric| Formula| Notes
MAE| mean(| y-ŷ
MSE| mean((y-ŷ)^2)| Penalizes large errors
RMSE| √MSE| Same units
R²| Explained variance| [-∞,1]
MAPE| % error| Undefined at y=0

---

🔁 Cross Validation

K-Fold:
1. Split into K folds
2. Train on K-1
3. Validate on 1
4. Repeat

---

🔢 Confusion Matrix

             Predicted
           P        N
Actual P   TP       FN
       N   FP       TN

---

⚙️ Hyperparameters

Parameter| Purpose| Range
max_depth| Controls depth| 3–10
min_samples_split| Min samples to split| 2–20
min_samples_leaf| Min samples in leaf| 1–10
max_features| Features per split| sqrt/log2

---

❓ Common Interview Questions

---

Q1: How do decision trees work?

Decision trees recursively split data using a greedy approach:

- Choose best feature (Gini/MSE)
- Split data
- Repeat until stopping criteria
- Leaf gives prediction

---

Q2: Advantages vs Disadvantages

✅ Advantages

- Interpretable
- No scaling needed
- Handles mixed data
- Non-linear

❌ Disadvantages

- Overfitting
- Instability
- Greedy decisions

---

Q3: Handling Overfitting

Pre-pruning

- max_depth
- min_samples_leaf

Post-pruning

- Cost complexity pruning

Ensemble

- Random Forest
- Boosting

---

Q4: Feature Importance

from sklearn.inspection import permutation_importance

# Built-in importance
model.feature_importances_

# Permutation importance
permutation_importance(model, X_test, y_test)

---

Q5: Visualization

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(model,
          feature_names=feature_names,
          class_names=class_names,
          filled=True)
plt.show()

---

Q6: Gini vs Entropy

Aspect| Gini| Entropy
Speed| Faster| Slower
Nature| Aggressive| Balanced
Use| Default| Info theory

---

Q7: Missing Values

- Imputation
- Surrogate splits
- Separate category

---

Q8: Bagging vs Boosting

Aspect| Bagging| Boosting
Training| Parallel| Sequential
Goal| Reduce variance| Reduce bias
Example| Random Forest| XGBoost

---

Q9: Complexity

- Training: O(n log n · m)
- Prediction: O(depth)
- Space: O(nodes)

---

Q10: Explain to Non-Tech

Use analogy:

«"It's like a flowchart asking questions step by step until it reaches a decision."»

---

🧠 Key Formulas

- Gini: "1 - Σ(pi²)"
- Entropy: "-Σ(pi log₂ pi)"
- Information Gain:
  "IG = H(parent) - Σ (nj/n) H(child)"

---

✅ Final Checklist

- Understand Gini & Entropy
- Practice coding trees
- Know hyperparameters
- Understand overfitting
- Be ready with real-world examples

---

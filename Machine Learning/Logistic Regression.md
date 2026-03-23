# 📘 Logistic Regression — Interview Preparation Guide

## 🎯 Context

This guide is designed for roles requiring strong knowledge of Logistic Regression in **Marketing Analytics (adaptable to Healthcare/Finance)**.

---

# 🔹 1. Overview of Logistic Regression

## 📌 Definition & Purpose

Logistic Regression is a **supervised classification algorithm** used to model the probability of a categorical outcome.

* Outputs probability between **0 and 1**
* Uses **sigmoid function** to map linear combinations → probability

[
P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}
]

👉 **Mental Model**:

* Linear Regression → predicts continuous values
* Logistic Regression → predicts probability of class membership

---

## 📌 Key Assumptions

| Assumption                   | Explanation                                          |
| ---------------------------- | ---------------------------------------------------- |
| Linearity in log-odds        | Relationship between features and log-odds is linear |
| Independence of observations | No duplicate or dependent samples                    |
| No multicollinearity         | Features should not be highly correlated             |
| Large sample size            | Needed for stable Maximum Likelihood Estimation      |
| No extreme outliers          | Outliers can distort coefficients                    |

---

## 📌 Types of Logistic Regression

| Type        | Description                     | Example                                 |
| ----------- | ------------------------------- | --------------------------------------- |
| Binary      | Two classes                     | Churn vs Not Churn                      |
| Multinomial | More than two unordered classes | Product category prediction             |
| Ordinal     | Ordered categories              | Customer satisfaction (Low/Medium/High) |

---

# 🔹 2. Practical Applications

## 📌 Use Cases (Marketing Focus)

| Problem               | Description                               |
| --------------------- | ----------------------------------------- |
| Customer Churn        | Predict if a user will leave              |
| Conversion Prediction | Will a user buy after ad exposure?        |
| Click-Through Rate    | Predict ad click probability              |
| Lead Scoring          | Rank leads based on likelihood to convert |

---

## 📌 Case Study (Conversion Prediction)

**Problem:** Predict whether a user will convert.

**Features:**

* User demographics
* Session duration
* Past purchase behavior
* Ad type

**Target:**

* 1 = Converted
* 0 = Not Converted

**Why Logistic Regression?**

* Interpretable (coefficients explain impact)
* Probabilistic output supports business decisions

---

# 🔹 3. Data Preparation

## 📌 Pipeline

```text
Raw Data → Cleaning → Encoding → Scaling → Feature Engineering → Model
```

---

## 📌 Key Steps

### 1. Handle Missing Values

* Drop or impute (mean/median/mode)

### 2. Encode Categorical Variables

* One-hot encoding (nominal)
* Label encoding (ordinal)

### 3. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
```

### 4. Train-Test Split

```python
from sklearn.model_selection import train_test_split
```

---

## 📌 Feature Selection

Why important:

* Improves model stability
* Reduces overfitting
* Enhances interpretability

Techniques:

* Correlation analysis
* VIF (Variance Inflation Factor)
* L1 regularization (Lasso)

---

## 📌 Transformations

| Issue         | Solution            |
| ------------- | ------------------- |
| Skewed data   | Log transform       |
| Non-linearity | Polynomial features |
| High variance | Scaling             |

---

# 🔹 4. Model Evaluation

## 📌 Confusion Matrix

|          | Predicted 1 | Predicted 0 |
| -------- | ----------- | ----------- |
| Actual 1 | TP          | FN          |
| Actual 0 | FP          | TN          |

---

## 📌 Metrics

| Metric    | Formula                             | When to Use             |
| --------- | ----------------------------------- | ----------------------- |
| Accuracy  | (TP+TN)/Total                       | Balanced datasets       |
| Precision | TP/(TP+FP)                          | False positives costly  |
| Recall    | TP/(TP+FN)                          | False negatives costly  |
| F1 Score  | Harmonic mean of Precision & Recall | Imbalanced datasets     |
| ROC-AUC   | Area under ROC curve                | Overall ranking ability |

---

## 📌 ROC Curve

* Plots TPR vs FPR
* AUC closer to 1 = better model

---

## 📌 Validation Techniques

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score
```

### Train/Validation/Test Split

* Prevents overfitting

---

# 🔹 5. Common Interview Questions

## 📌 Core Questions

### 1. Logistic vs Linear Regression

* Linear predicts continuous values
* Logistic predicts probability using sigmoid

---

### 2. Assumptions of Logistic Regression

* Linearity in log-odds
* Independence
* No multicollinearity
* Large sample size

---

### 3. Interpreting Coefficients

* Coefficients represent **log-odds change**
* Exponentiating gives **odds ratio**

Example:

* If β = 0.7 → odds increase by e^0.7 ≈ 2x

---

### 4. Handling Multicollinearity

* Remove correlated features
* Use VIF
* Apply regularization (L1/L2)

---

### 5. Model Performance Evaluation

* Confusion matrix
* Precision, Recall, F1
* ROC-AUC
* Cross-validation

---

## 📌 Advanced Questions

* Why use log-odds instead of probability directly?
* What is Maximum Likelihood Estimation (MLE)?
* Difference between L1 and L2 regularization?
* How does class imbalance affect logistic regression?
* How to choose decision threshold?
* When does logistic regression fail?

---

## 📌 Practical Coding Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

---

# 🧠 Key Takeaways (Interview Ready)

* Logistic regression = **probabilistic classifier**
* Interpretable → strong for business use cases
* Sensitive to multicollinearity and feature scaling
* Evaluation depends on business context, not just accuracy

---

# 🚀 Final Tip

In interviews, don’t just say *what logistic regression is*.
Explain:

* **When to use it**
* **Why it works**
* **Where it fails**

That’s the difference between "knows ML" and "hire this person immediately."

# 📘 Linear Regression – Complete Interview Preparation Guide

---

## 1. 🧠 Introduction

Linear Regression is one of the most fundamental supervised learning algorithms used for predicting a continuous target variable based on one or more input features.

### 🎯 Why it matters

* Simple, interpretable, and widely used baseline model
* Helps understand relationships between variables
* Foundation for many advanced models (GLMs, Neural Nets intuition)

### 📌 Intuition

Think of it as:

> "Finding the best-fitting straight line that minimizes prediction error"

---

## 2. 📐 Mathematical Formulation

### Simple Linear Regression

[
y = \beta_0 + \beta_1 x + \epsilon
]

### Multiple Linear Regression

[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon
]

### Matrix Form

[
y = X\beta + \epsilon
]

### Closed Form Solution (Normal Equation)

[
\hat{\beta} = (X^T X)^{-1} X^T y
]

---

## 3. ⚙️ Key Concepts

### 3.1 Assumptions (VERY IMPORTANT for interviews)

| Assumption           | Meaning                         | Violation Impact      |
| -------------------- | ------------------------------- | --------------------- |
| Linearity            | Relationship is linear          | Model underfits       |
| Independence         | Errors are independent          | Biased std errors     |
| Homoscedasticity     | Constant variance of errors     | Inefficient estimates |
| Normality of errors  | Errors are normally distributed | Affects inference     |
| No multicollinearity | Features not highly correlated  | Unstable coefficients |

---

### 3.2 Types of Linear Regression

#### 1. Simple Linear Regression

* One feature

#### 2. Multiple Linear Regression

* Multiple features

#### 3. Polynomial Regression

* Adds non-linearity using polynomial terms

#### 4. Regularized Regression

| Type       | Idea                 | When to Use       |
| ---------- | -------------------- | ----------------- |
| Ridge (L2) | Shrinks coefficients | Multicollinearity |
| Lasso (L1) | Feature selection    | Sparse models     |
| ElasticNet | Combination          | Balanced tradeoff |

---

### 3.3 Loss Function

Mean Squared Error (MSE):

[
MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
]

---

## 4. 📊 Evaluation Metrics

| Metric      | Formula                  | Use Case               |
| ----------- | ------------------------ | ---------------------- |
| MAE         | Mean absolute error      | Robust to outliers     |
| MSE         | Squared error            | Penalizes large errors |
| RMSE        | sqrt(MSE)                | Same unit as target    |
| R²          | Variance explained       | Model goodness         |
| Adjusted R² | Penalizes extra features | Feature selection      |

---

## 5. 🛠️ Training Methods

### 5.1 Closed Form (Normal Equation)

* Exact solution
* Expensive for large datasets

### 5.2 Gradient Descent

| Type       | Description  |
| ---------- | ------------ |
| Batch GD   | Full dataset |
| SGD        | One sample   |
| Mini-batch | Small chunks |

---

## 6. 🧪 Feature Engineering

* Scaling (important for GD, Ridge, Lasso)
* One-hot encoding
* Interaction terms
* Polynomial features

---

## 7. 🔍 Diagnostics & Debugging

* Residual plots
* QQ plots
* Variance Inflation Factor (VIF)
* Durbin-Watson test

---

## 8. 🌍 Real-World Use Cases

### 📦 Case Study 1: House Price Prediction

* Features: area, rooms, location
* Output: price

### 📈 Case Study 2: Sales Forecasting

* Features: ads spend, seasonality
* Output: revenue

### 🚗 Case Study 3: Mileage Prediction

* Features: engine size, weight
* Output: mileage

---

## 9. 💻 Code Implementation

### Using scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, preds))
print("R2:", r2_score(y_test, preds))
```

---

## 10. 🎯 Interview Questions & Answers

### 🔹 Theory Questions

**Q1. What is Linear Regression?**

* A supervised learning algorithm for predicting continuous values using a linear relationship.

**Q2. What assumptions does it make?**

* Linearity, independence, homoscedasticity, normality, no multicollinearity.

**Q3. What is multicollinearity?**

* High correlation between features causing unstable coefficients.

---

### 🔹 Application Questions

**Q4. When would you NOT use Linear Regression?**

* Non-linear relationships
* High outliers
* Non-constant variance

**Q5. How to handle multicollinearity?**

* Remove features
* Use Ridge/Lasso
* PCA

---

### 🔹 Coding Questions

**Q6. Implement Linear Regression from scratch**

```python
import numpy as np

class LinearRegressionScratch:
    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.theta
```

---

### 🔹 Advanced Questions

**Q7. Difference between Ridge and Lasso?**

* Ridge shrinks coefficients
* Lasso sets some to zero

**Q8. Why scale features?**

* Helps gradient descent converge faster

---

## 11. ⚠️ Common Pitfalls

* Ignoring assumptions
* Not checking residuals
* Using R² blindly
* Overfitting with too many features

---

## 12. 🧠 Pro Tips for Interviews

* Always explain intuition first
* Mention assumptions proactively
* Connect theory to real-world use cases
* Talk about limitations

---

## 🚀 Summary

Linear Regression is simple but extremely powerful. Interviewers use it to test your:

* Mathematical understanding
* Statistical intuition
* Practical ML skills

Master this = strong foundation for ML.

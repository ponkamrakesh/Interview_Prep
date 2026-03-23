# Linear Regression — Complete Interview Preparation Guide

---

## 1. Intuition First

Linear Regression tries to **fit a straight line (or hyperplane)** that best explains the relationship between input features (X) and target (y).

**Goal:**
Minimize the error between predicted and actual values.

👉 Think of it like:

> "Draw the best line through points so that overall error is as small as possible"

---

## 2. Mathematical Formulation

### Model

```
y = Xw + b
```

* y → target
* X → features
* w → weights
* b → bias

---

### Cost Function (OLS)

```
J(w) = (1/n) Σ (y_i - ŷ_i)^2
```

👉 Minimizes **Mean Squared Error (MSE)**

---

### Closed Form Solution (Normal Equation)

```
w = (XᵀX)^(-1) Xᵀy
```

⚠️ Works only when XᵀX is invertible

---

## 3. Assumptions (VERY IMPORTANT FOR INTERVIEWS)

### 1. Linearity

Relationship between X and y is linear

### 2. Independence

Observations are independent

### 3. Homoscedasticity

Constant variance of errors

### 4. Normality of Errors

Residuals are normally distributed

### 5. No Multicollinearity

Features should not be highly correlated

---

## 4. Training Methods

### 1. Normal Equation

* Direct solution
* Expensive for large data

### 2. Gradient Descent

```
w = w - α * ∇J(w)
```

Types:

* Batch
* Stochastic (SGD)
* Mini-batch

---

## 5. Feature Scaling

Required for Gradient Descent

* Standardization
* Normalization

👉 Without scaling → slow convergence

---

## 6. Regularization

### Why?

Prevent overfitting

---

### Ridge (L2)

```
J = MSE + λ Σ w²
```

* Shrinks weights
* No feature elimination

---

### Lasso (L1)

```
J = MSE + λ Σ |w|
```

* Can make weights zero
* Feature selection

---

### Elastic Net

Combination of L1 + L2

---

## 7. Evaluation Metrics

### 1. MSE

```
MSE = (1/n) Σ (y - ŷ)²
```

### 2. RMSE

```
RMSE = √MSE
```

### 3. MAE

```
MAE = (1/n) Σ |y - ŷ|
```

### 4. R² Score

```
R² = 1 - (SS_res / SS_tot)
```

---

## 8. Residual Analysis

Check assumptions using:

* Residual vs Fitted plot
* QQ plot
* Durbin-Watson test

---

## 9. Multicollinearity

Detection:

* VIF (Variance Inflation Factor)

```
VIF = 1 / (1 - R²)
```

Fix:

* Remove features
* PCA
* Ridge regression

---

## 10. Bias-Variance Tradeoff

* High Bias → Underfitting
* High Variance → Overfitting

Regularization helps balance

---

## 11. When to Use

✅ Linear relationships
✅ Interpretable models needed

❌ Non-linear patterns (unless transformed)

---

## 12. Common Mistakes

* Not checking assumptions
* Ignoring multicollinearity
* Using R² alone
* No feature scaling

---

## 13. Python Implementation

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = LinearRegression()
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# metrics
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
```

---

## 14. Advanced Topics

* Polynomial Regression
* Interaction Terms
* Weighted Least Squares
* Robust Regression (RANSAC)

---

# 🎯 Interview Questions (ALL Levels)

## Beginner

* What is Linear Regression?
* Difference between regression and classification?
* What is MSE?
* What is R² score?

---

## Intermediate

* Derive the normal equation
* Why do we square errors?
* What happens if assumptions are violated?
* Explain multicollinearity
* Why feature scaling?

---

## Advanced

* Derive gradient descent update rule
* Ridge vs Lasso?
* When will Lasso fail?
* What if XᵀX is not invertible?
* Explain bias-variance tradeoff

---

## Scenario-Based

* High R² but poor predictions. Why?
* Residuals show pattern. What to do?
* Features highly correlated. Fix?
* Outliers present. Solution?

---

## FAANG-Level / Deep Thinking

* Why is OLS unbiased?
* Geometric interpretation of Linear Regression
* Connection between Linear Regression and Maximum Likelihood
* Why Gaussian noise assumption?
* When does Linear Regression fail badly?

---

## Coding Questions

* Implement Linear Regression from scratch
* Implement Gradient Descent
* Add L2 regularization

---

## System Design Angle

* How to scale Linear Regression to big data?
* Online learning approach?

---

## Edge Cases

* Perfect multicollinearity
* Small dataset
* High dimensional data (p >> n)

---

## One-Liners for Interviews

* "Linear Regression minimizes squared error using OLS"
* "Ridge reduces variance, Lasso does feature selection"
* "Assumptions validate model reliability, not just performance"

---

🔥 Done. This is interview-ready, copy-paste and pretend you always knew all this.

# 📘 K Nearest Neighbors (KNN) — Interview Preparation Guide

## 🎯 Context

This guide is designed for Data Science interviews requiring strong understanding of **KNN (K-Nearest Neighbors)** with both theory and practical insights.

---

# 🔹 1. Overview of KNN

## 📌 Definition

K-Nearest Neighbors (KNN) is a **supervised learning algorithm** used for **classification and regression**.

* It is a **non-parametric** method (no explicit training phase)
* It is a **lazy learner** (stores data and computes at prediction time)

👉 **Mental Model**:
"Tell me who your neighbors are, and I’ll tell you who you are."

---

## 📌 Why KNN Matters

* Simple and intuitive
* No assumptions about data distribution
* Strong baseline model
* Useful in low-dimensional datasets

---

## 📌 Use Cases

| Domain                 | Example                                  |
| ---------------------- | ---------------------------------------- |
| Healthcare             | Disease classification based on symptoms |
| Finance                | Credit risk classification               |
| Marketing              | Customer segmentation                    |
| Recommendation Systems | Similar user/item suggestions            |

---

# 🔹 2. How KNN Works

## 📌 Algorithm Steps

1. Choose number of neighbors (K)
2. Compute distance between query point and all training points
3. Select K nearest neighbors
4. Aggregate:

   * Classification → majority vote
   * Regression → average of neighbors

---

## 📌 Distance Metrics

| Metric    | Formula                | When to Use       |   |                       |
| --------- | ---------------------- | ----------------- | - | --------------------- |
| Euclidean | √Σ(x₁ - x₂)²           | Continuous data   |   |                       |
| Manhattan | Σ                      | x₁ - x₂           |   | High-dimensional data |
| Minkowski | Generalized form       | Flexible choice   |   |                       |
| Cosine    | Angle-based similarity | Text / embeddings |   |                       |

---

## 📌 Key Parameters

### 1. K (Number of Neighbors)

* Small K → high variance (overfitting)
* Large K → high bias (underfitting)

👉 Rule of thumb:

* Start with √N

---

### 2. Distance Metric

* Impacts neighbor selection heavily

---

### 3. Weighting

* Uniform weights
* Distance-based weights (closer points matter more)

---

## 📌 Important Note: Feature Scaling

KNN is **distance-based**, so scaling is critical.

```python
from sklearn.preprocessing import StandardScaler
```

---

# 🔹 3. Practical Applications

## 📌 Classification Example

**Problem:** Spam detection

* Input: Email features
* Output: Spam / Not Spam

KNN predicts based on similarity to known emails.

---

## 📌 Regression Example

**Problem:** House price prediction

* Find K similar houses
* Average their prices

---

## 📌 Real-World Use Cases

| Application         | Description                 |
| ------------------- | --------------------------- |
| Image Recognition   | Similar image matching      |
| Recommender Systems | Find similar users/items    |
| Anomaly Detection   | Outliers far from neighbors |

---

# 🔹 4. Strengths & Weaknesses

## ✅ Strengths

* Simple to implement
* No training time
* Flexible (works for classification & regression)
* No assumptions about data

---

## ❌ Weaknesses

| Issue                   | Explanation                                     |
| ----------------------- | ----------------------------------------------- |
| Slow at prediction      | Must compute distance to all points             |
| Memory intensive        | Stores full dataset                             |
| Curse of dimensionality | Distance becomes meaningless in high dimensions |
| Sensitive to scaling    | Requires normalization                          |
| Sensitive to noise      | Outliers affect neighbors                       |

---

# 🔹 5. Common Interview Questions

## 📌 Conceptual Questions

* What is KNN and how does it work?
* Why is KNN called a lazy learner?
* Difference between KNN and logistic regression?
* How do you choose K?
* What happens if K is too small or too large?

---

## 📌 Practical Questions

* How do you handle large datasets in KNN?

  * KD-Trees / Ball Trees
  * Approximate Nearest Neighbors

* How do you deal with imbalanced data?

  * Weighted KNN

* How do you choose distance metric?

  * Based on data type and domain

---

## 📌 Coding Questions

### Basic Implementation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
```

---

### Advanced Questions

* Implement KNN from scratch
* Optimize KNN for large datasets
* Explain curse of dimensionality
* Compare KNN with SVM/Decision Trees

---

# 🧠 Key Takeaways

* KNN = **distance-based, lazy learning algorithm**
* Works well for small datasets
* Scaling is critical
* Suffers in high dimensions
* Simple but powerful baseline model

---

# 🚀 Interview Tip

Don’t just explain KNN.
Explain trade-offs:

* Why it works
* When it fails
* How to optimize it

That’s what actually gets you hired.

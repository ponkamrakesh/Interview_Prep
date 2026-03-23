🧠 1. Regression Metrics
🎯 Core Idea

We measure how far predictions are from actual values.

📊 Key Metrics
1. Mean Absolute Error (MAE)
𝑀
𝐴
𝐸
=
1
𝑛
∑
∣
𝑦
𝑖
−
𝑦
^
𝑖
∣
MAE=
n
1
	​

∑∣y
i
	​

−
y
^
	​

i
	​

∣
Intuition: Average absolute mistake
Robust to outliers (linear penalty)
Use when: You want interpretability (same unit as target)
2. Mean Squared Error (MSE)
𝑀
𝑆
𝐸
=
1
𝑛
∑
(
𝑦
𝑖
−
𝑦
^
𝑖
)
2
MSE=
n
1
	​

∑(y
i
	​

−
y
^
	​

i
	​

)
2
Penalizes large errors heavily
Use when: Outliers matter a lot
3. Root Mean Squared Error (RMSE)
𝑅
𝑀
𝑆
𝐸
=
𝑀
𝑆
𝐸
RMSE=
MSE
	​

Same unit as target
Most commonly used
4. R² Score (Coefficient of Determination)
𝑅
2
=
1
−
𝑆
𝑆
𝑟
𝑒
𝑠
𝑆
𝑆
𝑡
𝑜
𝑡
R
2
=1−
SS
tot
	​

SS
res
	​

	​

Measures explained variance
Range: (-∞, 1)
Use when: Model comparison
5. Adjusted R²
𝑅
𝑎
𝑑
𝑗
2
=
1
−
(
1
−
𝑅
2
)
𝑛
−
1
𝑛
−
𝑝
−
1
R
adj
2
	​

=1−(1−R
2
)
n−p−1
n−1
	​

Penalizes useless features
Use when: Multiple features
6. Mean Absolute Percentage Error (MAPE)
𝑀
𝐴
𝑃
𝐸
=
100
𝑛
∑
∣
𝑦
𝑖
−
𝑦
^
𝑖
𝑦
𝑖
∣
MAPE=
n
100
	​

∑
	​

y
i
	​

y
i
	​

−
y
^
	​

i
	​

	​

	​

Percentage error
Avoid: when 
𝑦
𝑖
≈
0
y
i
	​

≈0
💻 Code
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np

y_true = [100, 200, 300]
y_pred = [110, 190, 310]

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(mae, mse, rmse, r2)
⚠️ Common Mistakes
Using RMSE when business cares about absolute error
Using MAPE with zeros
Trusting R² blindly (can be negative 👀)
🎯 2. Classification Metrics
🎯 Core Idea

We measure how well classes are predicted.

🧩 Confusion Matrix
	Pred +	Pred -
Actual +	TP	FN
Actual -	FP	TN
📊 Key Metrics
1. Accuracy
𝐴
𝑐
𝑐
𝑢
𝑟
𝑎
𝑐
𝑦
=
𝑇
𝑃
+
𝑇
𝑁
𝑇
𝑜
𝑡
𝑎
𝑙
Accuracy=
Total
TP+TN
	​

Bad for imbalanced data
2. Precision
𝑃
𝑟
𝑒
𝑐
𝑖
𝑠
𝑖
𝑜
𝑛
=
𝑇
𝑃
𝑇
𝑃
+
𝐹
𝑃
Precision=
TP+FP
TP
	​

“When model says YES, how often correct?”
Use in spam / fraud
3. Recall (Sensitivity)
𝑅
𝑒
𝑐
𝑎
𝑙
𝑙
=
𝑇
𝑃
𝑇
𝑃
+
𝐹
𝑁
Recall=
TP+FN
TP
	​

“How many positives did we catch?”
Use in medical / fraud
4. F1 Score
𝐹
1
=
2
⋅
𝑃
𝑟
𝑒
𝑐
𝑖
𝑠
𝑖
𝑜
𝑛
⋅
𝑅
𝑒
𝑐
𝑎
𝑙
𝑙
𝑃
𝑟
𝑒
𝑐
𝑖
𝑠
𝑖
𝑜
𝑛
+
𝑅
𝑒
𝑐
𝑎
𝑙
𝑙
F1=2⋅
Precision+Recall
Precision⋅Recall
	​

Balance between precision & recall
5. Specificity
𝑆
𝑝
𝑒
𝑐
𝑖
𝑓
𝑖
𝑐
𝑖
𝑡
𝑦
=
𝑇
𝑁
𝑇
𝑁
+
𝐹
𝑃
Specificity=
TN+FP
TN
	​

6. ROC-AUC
Area under ROC curve
Threshold independent
7. Log Loss (Cross Entropy)
−
1
𝑛
∑
𝑦
log
⁡
(
𝑝
)
+
(
1
−
𝑦
)
log
⁡
(
1
−
𝑝
)
−
n
1
	​

∑ylog(p)+(1−y)log(1−p)
Penalizes confident wrong predictions heavily
💻 Code
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
y_prob = [0.1, 0.9, 0.4, 0.2]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_prob))
print("LogLoss:", log_loss(y_true, y_prob))
⚠️ When to Use What
Scenario	Metric
Balanced data	Accuracy
Fraud detection	Recall
Spam filtering	Precision
Imbalanced data	F1 / ROC-AUC
Probabilistic models	Log Loss
🔵 3. Clustering Metrics
🎯 Core Idea

No labels. So we measure structure quality.

📊 Key Metrics
1. Silhouette Score
𝑠
=
𝑏
−
𝑎
max
⁡
(
𝑎
,
𝑏
)
s=
max(a,b)
b−a
	​

a = intra-cluster distance
b = nearest cluster distance
Range: [-1, 1]
2. Davies-Bouldin Index (DBI)
𝐷
𝐵
=
1
𝑘
∑
max
⁡
𝜎
𝑖
+
𝜎
𝑗
𝑑
(
𝑐
𝑖
,
𝑐
𝑗
)
DB=
k
1
	​

∑max
d(c
i
	​

,c
j
	​

)
σ
i
	​

+σ
j
	​

	​

Lower is better
3. Calinski-Harabasz Index
𝐵
𝑒
𝑡
𝑤
𝑒
𝑒
𝑛
 
𝐶
𝑙
𝑢
𝑠
𝑡
𝑒
𝑟
 
𝑉
𝑎
𝑟
𝑖
𝑎
𝑛
𝑐
𝑒
𝑊
𝑖
𝑡
ℎ
𝑖
𝑛
 
𝐶
𝑙
𝑢
𝑠
𝑡
𝑒
𝑟
 
𝑉
𝑎
𝑟
𝑖
𝑎
𝑛
𝑐
𝑒
Within Cluster Variance
Between Cluster Variance
	​

Higher is better
4. Inertia (K-Means)
∑
∣
∣
𝑥
−
𝑐
∣
∣
2
∑∣∣x−c∣∣
2
Used in elbow method
5. Adjusted Rand Index (ARI)
Requires ground truth
Measures clustering similarity
💻 Code
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

X = [[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]]

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

score = silhouette_score(X, labels)
print(score)
⚠️ Mistakes
Using accuracy for clustering (why would you do that…)
Ignoring scaling before clustering
🟣 4. Dimensionality Reduction Metrics
🎯 Core Idea

Check if information is preserved

📊 Key Metrics
1. Explained Variance (PCA)
𝜆
𝑖
∑
𝜆
∑λ
λ
i
	​

	​

How much variance each component captures
2. Reconstruction Error
∣
∣
𝑋
−
𝑋
𝑟
𝑒
𝑐
𝑜
𝑛
𝑠
𝑡
𝑟
𝑢
𝑐
𝑡
𝑒
𝑑
∣
∣
2
∣∣X−X
reconstructed
	​

∣∣
2
Lower = better
3. Trustworthiness (t-SNE, UMAP)
Measures local structure preservation
4. KL Divergence (t-SNE)
Difference between distributions
💻 Code
from sklearn.decomposition import PCA
import numpy as np

X = np.random.rand(100, 5)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("Explained Variance:", pca.explained_variance_ratio_)
⚠️ When to Use What
Technique	Metric
PCA	Explained variance
Autoencoder	Reconstruction error
t-SNE	KL divergence
UMAP	Trustworthiness
🧩 Final Mental Model (The One You Actually Remember)
Regression → distance between numbers
Classification → correctness of labels
Clustering → structure quality
Dimensionality Reduction → information preservation

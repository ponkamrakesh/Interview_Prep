# 🌳 Machine Learning, Data Science & Generative AI Interview Knowledge Tree

> A comprehensive preparation guide covering 300+ concepts across Beginner → Intermediate → Senior levels

---

# 1. Machine Learning

## 1.1 Supervised Learning

### 1.1.1 Regression
- **Linear Regression**
  - Ordinary Least Squares (OLS) — minimizes sum of squared residuals
  - Assumptions: linearity, independence, homoscedasticity, normality, no multicollinearity
  - Closed-form solution: θ = (XᵀX)⁻¹Xᵀy
  - Gradient descent variants: batch, stochastic, mini-batch
  - Learning rate selection and convergence criteria
  - Feature scaling necessity for gradient descent
- **Regularized Regression**
  - Ridge (L2): adds λΣθ² — shrinks coefficients, handles multicollinearity
  - Lasso (L1): adds λΣ|θ| — induces sparsity, feature selection
  - Elastic Net: combines L1 + L2 penalties
  - Comparison: when to use which (correlated features → Ridge, sparse solutions → Lasso)
- **Polynomial Regression**
  - Basis function expansion
  - Degree selection and bias-variance tradeoff
  - Overfitting risks with high-degree polynomials
- **Robust Regression**
  - Huber loss: quadratic near zero, linear for large errors
  - RANSAC: random sample consensus for outlier handling
  - Theil-Sen estimator: median-based slope estimation

### 1.1.2 Classification
- **Logistic Regression**
  - Sigmoid function: σ(z) = 1/(1+e⁻ᶻ)
  - Log-odds interpretation
  - Maximum likelihood estimation
  - Multinomial/Softmax extension for multi-class
  - Decision boundary: linear in feature space
- **Support Vector Machines (SVM)**
  - Maximum margin principle
  - Hard margin vs soft margin (C parameter)
  - Kernel trick: linear, polynomial, RBF, sigmoid
  - Support vectors and their role
  - Hinge loss formulation
  - SMO algorithm (Sequential Minimal Optimization)
  - Multi-class strategies: one-vs-one, one-vs-rest
- **Decision Trees**
  - Splitting criteria: Gini impurity, entropy/information gain, variance reduction
  - ID3, C4.5, CART algorithms
  - Tree pruning: pre-pruning, post-pruning (cost-complexity)
  - Handling missing values
  - Feature importance calculation
- **Ensemble Methods**
  - **Bagging**
    - Bootstrap sampling with replacement
    - Random Forest: bagging + feature randomness
    - Out-of-bag (OOB) error estimation
    - Feature importance: Gini importance, permutation importance
  - **Boosting**
    - AdaBoost: adaptive weighting of misclassified samples
    - Gradient Boosting: additive models fit to pseudo-residuals
    - XGBoost: regularized gradient boosting (L1/L2, tree pruning)
    - LightGBM: histogram-based, leaf-wise growth
    - CatBoost: ordered boosting, native categorical handling
    - Learning rate (shrinkage) and n_estimators tradeoff
  - **Stacking**
    - Meta-learner concept
    - Base model diversity importance
    - Blending vs stacking
- **Naive Bayes**
  - Bayes theorem application
  - Conditional independence assumption ("naive")
  - Gaussian, Multinomial, Bernoulli variants
  - Laplace smoothing for zero probabilities
  - Text classification applications
- **K-Nearest Neighbors (KNN)**
  - Distance metrics: Euclidean, Manhattan, Minkowski, Cosine
  - K selection: odd for binary, cross-validation
  - Curse of dimensionality impact
  - KD-trees and Ball-trees for efficient search
  - Weighted KNN: inverse distance weighting

### 1.1.3 Evaluation Metrics
- **Classification Metrics**
  - Confusion matrix: TP, FP, TN, FN
  - Accuracy: (TP+TN)/(total) — misleading with imbalanced data
  - Precision: TP/(TP+FP) — "of predicted positive, how many correct"
  - Recall/Sensitivity: TP/(TP+FN) — "of actual positive, how many found"
  - F1-Score: harmonic mean of precision and recall
  - F-beta: weighted harmonic mean (β controls precision/recall emphasis)
  - Specificity: TN/(TN+FP) — true negative rate
  - ROC-AUC: area under receiver operating characteristic curve
  - PR-AUC: area under precision-recall curve (better for imbalance)
  - Log-loss/cross-entropy: probabilistic evaluation
  - Cohen's Kappa: agreement beyond chance
  - Matthews Correlation Coefficient (MCC): balanced measure for imbalanced data
- **Regression Metrics**
  - MAE: mean absolute error — robust to outliers
  - MSE: mean squared error — penalizes large errors
  - RMSE: root MSE — same units as target
  - MSLE: mean squared log error — for exponential targets
  - MAPE: mean absolute percentage error — scale-independent
  - R²: coefficient of determination — proportion variance explained
  - Adjusted R²: penalizes for extra predictors

### 1.1.4 Imbalanced Data Handling
- **Resampling Techniques**
  - Random oversampling: duplicate minority samples (risk: overfitting)
  - Random undersampling: remove majority samples (risk: information loss)
  - SMOTE: Synthetic Minority Over-sampling Technique (interpolate in feature space)
  - ADASYN: adaptive SMOTE based on difficulty
  - Borderline-SMOTE: focus on decision boundary
  - Tomek links: remove overlapping pairs
  - Edited Nearest Neighbors: clean noisy majority samples
- **Algorithm-Level Approaches**
  - Class weights: inverse frequency weighting
  - Cost-sensitive learning: assign different misclassification costs
  - Threshold moving: optimize decision threshold post-training
  - Focal loss: down-weight easy examples

## 1.2 Unsupervised Learning

### 1.2.1 Clustering
- **K-Means**
  - Lloyd's algorithm: assign, update centroids iteratively
  - K selection: elbow method, silhouette score, gap statistic
  - K-means++: smart initialization
  - Limitations: spherical clusters, sensitive to outliers
  - Mini-batch K-means: scalable version
- **Hierarchical Clustering**
  - Agglomerative (bottom-up) vs Divisive (top-down)
  - Linkage criteria: single, complete, average, Ward
  - Dendrogram interpretation
  - Computational complexity O(n³) or O(n²logn)
- **Density-Based Clustering**
  - DBSCAN: core points, border points, noise
  - Parameters: eps (neighborhood radius), minPts
  - OPTICS: handles varying densities
  - HDBSCAN: hierarchical density-based
- **Gaussian Mixture Models (GMM)**
  - Soft clustering via probability assignment
  - EM algorithm: E-step (responsibilities), M-step (parameter updates)
  - BIC/AIC for component selection
  - Covariance types: spherical, diagonal, tied, full
- **Spectral Clustering**
  - Graph Laplacian construction
  - Eigenvalue decomposition
  - K-means on spectral embeddings
  - Affinity matrix construction

### 1.2.2 Dimensionality Reduction
- **Principal Component Analysis (PCA)**
  - Variance maximization interpretation
  - Eigenvalue decomposition of covariance matrix
  - SVD approach for numerical stability
  - Explained variance ratio
  - Number of components selection
  - Limitations: linear, ignores labels
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
  - Preserve local neighborhoods
  - Perplexity parameter: effective number of neighbors
  - Non-linear, stochastic
  - Not for generalization — visualization only
  - Computational cost O(n²)
- **UMAP (Uniform Manifold Approximation and Projection)**
  - Preserves global and local structure
  - Faster than t-SNE
  - Hyperparameters: n_neighbors, min_dist
  - Can be used for transformation (unlike t-SNE)
- **Linear Discriminant Analysis (LDA)**
  - Supervised dimensionality reduction
  - Maximize between-class variance, minimize within-class variance
  - Max (C-1) components where C = number of classes
- **Autoencoders for Dimensionality Reduction**
  - Encoder-decoder architecture
  - Bottleneck layer learns compressed representation
  - Denoising autoencoders for robustness

### 1.2.3 Anomaly Detection
- **Statistical Methods**
  - Z-score: values beyond 3σ
  - IQR method: Q1 - 1.5×IQR, Q3 + 1.5×IQR
  - Mahalanobis distance: accounts for correlations
- **Isolation Forest**
  - Random splits isolate anomalies faster
  - Average path length as anomaly score
  - Sub-sampling for efficiency
- **One-Class SVM**
  - Learn boundary of normal data
  - Nu parameter controls outlier fraction
- **Local Outlier Factor (LOF)**
  - Compare local density to neighbors' densities
  - k-distance and reachability distance

## 1.3 Model Selection & Validation

### 1.3.1 Cross-Validation
- **K-Fold Cross-Validation**
  - K selection: typically 5 or 10
  - Stratified K-Fold: preserve class distribution
  - Group K-Fold: keep related samples together
  - Time Series Split: respect temporal order
- **Leave-One-Out (LOO)**
  - N folds for N samples
  - Unbiased but high variance
  - Computationally expensive
- **Nested Cross-Validation**
  - Outer loop: model evaluation
  - Inner loop: hyperparameter tuning
  - Prevents optimistic bias in hyperparameter selection

### 1.3.2 Hyperparameter Tuning
- **Grid Search**
  - Exhaustive search over parameter grid
  - Computational cost grows exponentially
- **Random Search**
  - Sample from parameter distributions
  - More efficient than grid search for high dimensions
- **Bayesian Optimization**
  - Surrogate model (Gaussian Process, TPE)
  - Acquisition function: EI, PI, UCB
  - Sequential model-based optimization
- **Evolutionary Algorithms**
  - Genetic algorithms for hyperparameter search
  - Population-based training

### 1.3.3 Bias-Variance Tradeoff
- **Bias**
  - Systematic error from wrong assumptions
  - High bias → underfitting
  - Solutions: more complex model, more features
- **Variance**
  - Sensitivity to training data fluctuations
  - High variance → overfitting
  - Solutions: regularization, more data, simpler model
- **Irreducible Error**
  - Noise inherent in data
  - Cannot be eliminated
- **Decomposition**
  - MSE = Bias² + Variance + Irreducible Error

### 1.3.4 Learning Curves
- Training vs validation error vs training size
- Diagnosing underfitting (high bias)
- Diagnosing overfitting (high variance)
- Determining if more data will help

## 1.4 Feature Engineering

### 1.4.1 Feature Scaling
- **Standardization (Z-score)**
  - (x - μ) / σ
  - Centers at 0, unit variance
  - Required for distance-based algorithms
- **Min-Max Scaling**
  - (x - min) / (max - min)
  - Scales to [0, 1] range
  - Sensitive to outliers
- **Robust Scaling**
  - Uses median and IQR
  - Outlier-resistant
- **MaxAbs Scaling**
  - Scales by maximum absolute value
  - Preserves sparsity

### 1.4.2 Categorical Encoding
- **One-Hot Encoding**
  - Binary columns for each category
  - Dummy variable trap: drop one category
  - High cardinality issues
- **Label Encoding**
  - Integer assignment
  - Implies ordinal relationship (often wrong)
- **Ordinal Encoding**
  - For truly ordered categories
- **Target Encoding / Mean Encoding**
  - Replace category with mean target value
  - Risk of overfitting — use smoothing
- **Frequency Encoding**
  - Replace with occurrence count/frequency
- **Embeddings**
  - Learned representations for high cardinality

### 1.4.3 Feature Selection
- **Filter Methods**
  - Correlation with target
  - Mutual information
  - Chi-squared test for categorical features
  - ANOVA F-test
  - Variance threshold
- **Wrapper Methods**
  - Forward selection: start empty, add best
  - Backward elimination: start full, remove worst
  - Recursive Feature Elimination (RFE)
  - Computationally expensive
- **Embedded Methods**
  - Lasso regularization
  - Tree-based feature importance
  - Permutation importance

### 1.4.4 Feature Creation
- **Polynomial Features**
  - Interaction terms
  - Higher-order terms
- **Binning/Discretization**
  - Equal-width bins
  - Equal-frequency bins
  - Target-based bins
- **Date/Time Features**
  - Extract hour, day, month, year
  - Cyclical encoding: sin/cos for periodic features
  - Time since event
- **Text Features**
  - Character count, word count
  - TF-IDF vectors
  - Sentiment scores
- **Domain-Specific Features**
  - Ratios and differences
  - Aggregations (groupby operations)

## 1.5 Time Series

### 1.5.1 Time Series Components
- **Trend**
  - Long-term direction
  - Linear, exponential, logistic trends
- **Seasonality**
  - Regular periodic patterns
  - Additive vs multiplicative
- **Cyclical**
  - Long-term cycles without fixed period
- **Residual/Noise**
  - Irregular fluctuations

### 1.5.2 Stationarity
- **Definition**
  - Constant mean, variance, autocovariance over time
- **Tests**
  - Augmented Dickey-Fuller (ADF)
  - Kwiatkowski-Phillips-Schmidt-Shin (KPSS)
  - Phillips-Perron
- **Transformations**
  - Differencing (first-order, seasonal)
  - Log transformation for variance stabilization
  - Box-Cox transformation

### 1.5.3 Classical Models
- **ARIMA**
  - AR(p): autoregressive component
  - I(d): differencing for stationarity
  - MA(q): moving average component
  - ACF/PACF for parameter selection
  - AIC/BIC for model selection
- **SARIMA**
  - Seasonal ARIMA extension
  - Additional seasonal parameters (P, D, Q, m)
- **Exponential Smoothing**
  - Simple: no trend/seasonality
  - Holt's: with trend
  - Holt-Winters: with trend and seasonality
  - Smoothing parameters (α, β, γ)

### 1.5.4 Advanced Time Series
- **Prophet**
  - Additive regression model
  - Piecewise linear trend with changepoints
  - Fourier series for seasonality
  - Holiday effects
- **Vector Autoregression (VAR)**
  - Multivariate time series
  - Granger causality
- **State Space Models**
  - Kalman filter
  - Hidden Markov Models
- **Deep Learning for Time Series**
  - LSTM/GRU for sequences
  - Temporal Convolutional Networks
  - Transformers for time series

### 1.5.5 Time Series Evaluation
- **Cross-Validation**
  - Walk-forward validation
  - Expanding window
  - Sliding window
- **Metrics**
  - MAE, RMSE, MAPE
  - MASE: mean absolute scaled error
  - SMAPE: symmetric MAPE

### 🔴 Commonly Missed / Underrated Topics (ML)

- **Data Leakage**
  - Train-test contamination through preprocessing
  - Target leakage: features not available at prediction time
  - Temporal leakage: using future information
  - Prevention strategies
- **Calibration**
  - Reliability diagrams
  - Platt scaling (sigmoid calibration)
  - Isotonic regression
  - Expected Calibration Error (ECE)
  - Importance for probabilistic decisions
- **Concept Drift**
  - Sudden vs gradual drift
  - Detection methods: ADWIN, Page-Hinkley
  - Model retraining strategies
- **Causal Inference in ML**
  - Confounding variables
  - Propensity score matching
  - Causal forests
  - Difference-in-differences
- **Multiple Comparisons Problem**
  - Bonferroni correction
  - False Discovery Rate (FDR)
  - Family-wise error rate
- **Model Interpretability vs Performance Tradeoff**
  - When to prioritize interpretability
  - Regulatory requirements (GDPR "right to explanation")
- **Survival Analysis**
  - Censoring (right, left, interval)
  - Kaplan-Meier estimator
  - Cox proportional hazards
  - Time-varying covariates
- **Multi-Label Classification**
  - Label correlations
  - Binary relevance vs classifier chains
  - Label powerset
- **Cost-Sensitive Learning**
  - Asymmetric misclassification costs
  - Example-dependent costs
- **Active Learning**
  - Uncertainty sampling
  - Query-by-committee
  - Expected model change

### 🎯 Interview Focus (ML)

**Beginner Level:**
- Explain bias-variance tradeoff with examples
- When to use precision vs recall
- Difference between L1 and L2 regularization
- How random forest prevents overfitting
- Explain cross-validation and why it's needed

**Intermediate Level:**
- Derive gradient descent update for logistic regression
- Compare XGBoost vs LightGBM vs CatBoost
- How to handle 99:1 class imbalance
- Explain ROC-AUC vs PR-AUC tradeoffs
- Feature selection strategies for 10K+ features

**Senior Level:**
- Design an end-to-end ML system with monitoring
- Handle concept drift in production
- Optimize inference latency for tree ensemble
- Design A/B test for model deployment
- Address fairness and bias in ML models

**Differentiating Questions:**
- "Your model performs well offline but poorly online — what could be wrong?" (Answer: data leakage, distribution shift)
- "How would you detect if your model is becoming stale?" (Answer: monitoring prediction distributions, performance degradation)
- "Design a recommendation system that handles cold start" (Answer: content-based → collaborative filtering hybrid)

---

# 2. Deep Learning

## 2.1 Neural Network Fundamentals

### 2.1.1 Perceptron and MLP
- **Perceptron**
  - Weighted sum + activation
  - Linear decision boundary
  - Perceptron convergence theorem
- **Multi-Layer Perceptron (MLP)**
  - Hidden layers enable non-linear separation
  - Universal approximation theorem
  - Forward propagation: matrix multiplications
  - Backpropagation: chain rule application

### 2.1.2 Activation Functions
- **Sigmoid**
  - σ(x) = 1/(1+e⁻ˣ)
  - Output range (0, 1)
  - Vanishing gradient problem
  - Not zero-centered
- **Tanh**
  - Output range (-1, 1)
  - Zero-centered but still vanishing gradient
- **ReLU (Rectified Linear Unit)**
  - f(x) = max(0, x)
  - Solves vanishing gradient
  - Dying ReLU problem (negative inputs)
  - Computationally efficient
- **Leaky ReLU / PReLU**
  - Small negative slope (αx for x < 0)
  - Prevents dying ReLU
- **ELU (Exponential Linear Unit)**
  - Smooth negative region
  - Mean closer to zero
- **Swish / SiLU**
  - f(x) = x · σ(x)
  - Self-gated, smooth
  - Used in some transformer architectures
- **Softmax**
  - Multi-class probability distribution
  - Numerical stability: subtract max before exp

### 2.1.3 Weight Initialization
- **Xavier/Glorot Initialization**
  - Var(W) = 2/(fan_in + fan_out)
  - Good for sigmoid/tanh
- **He Initialization**
  - Var(W) = 2/fan_in
  - Designed for ReLU
- **Orthogonal Initialization**
  - Preserves norm through layers
- **Impact of Poor Initialization**
  - Vanishing/exploding gradients
  - Symmetry breaking

### 2.1.4 Normalization Techniques
- **Batch Normalization**
  - Normalize per batch: (x - μ_batch) / √(σ²_batch + ε)
  - Learnable scale (γ) and shift (β)
  - Reduces internal covariate shift
  - Acts as regularization
  - Inference: use running statistics
- **Layer Normalization**
  - Normalize across features (per sample)
  - Independent of batch size
  - Used in RNNs and Transformers
- **Instance Normalization**
  - Normalize per channel per sample
  - Style transfer applications
- **Group Normalization**
  - Middle ground between batch and instance
  - Effective for small batches
- **Weight Normalization**
  - Decouple magnitude and direction

### 2.1.5 Regularization
- **L1/L2 Regularization**
  - Weight decay in optimizers
- **Dropout**
  - Randomly zero neurons during training
  - Inverted dropout for scaling
  - Dropout rate selection (typically 0.2-0.5)
  - Monte Carlo dropout for uncertainty
- **Early Stopping**
  - Monitor validation loss
  - Restore best weights
- **Data Augmentation**
  - Transform training data
  - Domain-specific augmentations
- **Label Smoothing**
  - Soft targets: 0.9 instead of 1
  - Prevents overconfident predictions

## 2.2 Optimization

### 2.2.1 Gradient Descent Variants
- **Batch Gradient Descent**
  - Full dataset each step
  - Stable but slow
- **Stochastic Gradient Descent (SGD)**
  - One sample per step
  - Noisy but faster per iteration
  - Can escape local minima
- **Mini-batch Gradient Descent**
  - Compromise: typically 32-512 samples
  - Vectorized operations
  - Batch size tradeoffs

### 2.2.2 Optimization Algorithms
- **Momentum**
  - Accumulate velocity: v_t = βv_{t-1} + ∇L
  - Accelerates in consistent directions
  - Dampens oscillations
- **Nesterov Accelerated Gradient (NAG)**
  - Lookahead gradient
  - More responsive
- **AdaGrad**
  - Per-parameter learning rates
  - Accumulate squared gradients
  - Learning rate decay too aggressive
- **RMSprop**
  - Exponential moving average of squared gradients
  - Fixes AdaGrad's aggressive decay
- **Adam (Adaptive Moment Estimation)**
  - Combines momentum + RMSprop
  - First moment (mean) + second moment (uncentered variance)
  - Bias correction for early iterations
  - Default choice for many problems
- **AdamW**
  - Decoupled weight decay
  - Better regularization than Adam+L2
- **Learning Rate Schedules**
  - Step decay: reduce at epochs
  - Exponential decay
  - Cosine annealing
  - Warm restarts
  - One-cycle policy

### 2.2.3 Training Challenges
- **Vanishing Gradients**
  - Gradients become very small in deep networks
  - Solutions: ReLU, skip connections, careful initialization
- **Exploding Gradients**
  - Gradients become very large
  - Solutions: gradient clipping
- **Dead Neurons**
  - ReLU neurons that never activate
  - Solutions: Leaky ReLU, proper initialization
- **Saddle Points**
  - More common than local minima in high dimensions
  - SGD noise helps escape

## 2.3 Convolutional Neural Networks (CNNs)

### 2.3.1 Convolution Operations
- **2D Convolution**
  - Kernel slides over input
  - Output size: (W - K + 2P)/S + 1
  - Parameters: kernel size, stride, padding
- **Padding**
  - Valid (no padding): spatial reduction
  - Same: preserve spatial dimensions
  - Dilated/atrous: expand receptive field
- **Stride**
  - Controls output spatial size
  - Downsampling effect
- **Receptive Field**
  - Input region affecting output
  - Grows with depth

### 2.3.2 Pooling Layers
- **Max Pooling**
  - Take maximum in window
  - Translation invariance
  - Most common choice
- **Average Pooling**
  - Take average in window
  - Smoother, less common now
- **Global Pooling**
  - Pool entire feature map
  - Reduces to vector

### 2.3.3 CNN Architectures
- **LeNet-5**
  - 1998, foundational
- **AlexNet**
  - 2012, ReLU + dropout + GPU
- **VGGNet**
  - 3×3 convolutions, deep and uniform
- **ResNet**
  - Skip connections (residual blocks)
  - Enables very deep networks (152+ layers)
  - Identity mapping when F(x) = 0
- **Inception**
  - Multi-scale convolutions
  - 1×1 convolutions for dimension reduction
- **EfficientNet**
  - Compound scaling: depth, width, resolution
  - Mobile-first design
- **Vision Transformers (ViT)**
  - Patch embeddings + transformer
  - Global attention vs local CNN

### 2.3.4 Advanced CNN Techniques
- **Batch Normalization in CNNs**
  - Normalize per channel
- **Transfer Learning**
  - Pre-trained features
  - Fine-tuning strategies
- **Data Augmentation for Images**
  - Flip, rotate, crop, color jitter
  - Mixup, CutMix
  - AutoAugment, RandAugment

## 2.4 Recurrent Neural Networks (RNNs)

### 2.4.1 Basic RNN
- **Architecture**
  - Hidden state: h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b)
  - Shared weights across time
- **Limitations**
  - Vanishing gradients in long sequences
  - Short-term memory only

### 2.4.2 LSTM (Long Short-Term Memory)
- **Cell State**
  - Conveyor belt for information
- **Gates**
  - Forget gate: what to discard
  - Input gate: what to store
  - Output gate: what to output
- **Equations**
  - f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
  - i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
  - C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
  - C_t = f_t * C_{t-1} + i_t * C̃_t
  - o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
  - h_t = o_t * tanh(C_t)

### 2.4.3 GRU (Gated Recurrent Unit)
- **Simplified LSTM**
  - Merge cell and hidden state
  - Update gate + reset gate
  - Fewer parameters, faster

### 2.4.4 RNN Variants
- **Bidirectional RNN**
  - Process sequence both directions
  - Future context available
- **Stacked RNNs**
  - Multiple RNN layers
- **Encoder-Decoder (Seq2Seq)**
  - Encoder: compress to context vector
  - Decoder: generate output sequence
  - Teacher forcing during training

## 2.5 Attention Mechanisms

### 2.5.1 Attention Fundamentals
- **Core Idea**
  - Focus on relevant parts of input
  - Query-Key-Value framework
- **Score Functions**
  - Dot product: Q·Kᵀ
  - Scaled dot product: (Q·Kᵀ)/√d_k
  - Additive (Bahdanau): vᵀtanh(W_q·Q + W_k·K)

### 2.5.2 Self-Attention
- **Intra-sequence attention**
  - Each position attends to all positions
  - Captures long-range dependencies
- **Computational Complexity**
  - O(n²) in sequence length
  - Memory bottleneck for long sequences

### 2.5.3 Multi-Head Attention
- **Parallel attention heads**
  - Different representation subspaces
  - Concatenate and project
- **Benefits**
  - Multiple attention patterns
  - Increased expressiveness

### 2.5.4 Cross-Attention
- **Between two sequences**
  - Query from one, Key-Value from another
  - Used in encoder-decoder

## 2.6 Generative Models

### 2.6.1 Autoencoders
- **Architecture**
  - Encoder: input → latent
  - Decoder: latent → reconstruction
- **Loss**
  - Reconstruction loss (MSE, BCE)
- **Applications**
  - Denoising, anomaly detection

### 2.6.2 Variational Autoencoders (VAE)
- **Latent Space**
  - Probabilistic: learn distribution parameters (μ, σ)
- **Reparameterization Trick**
  - z = μ + σ·ε, where ε ~ N(0,1)
  - Enables backpropagation through sampling
- **ELBO (Evidence Lower Bound)**
  - Reconstruction + KL divergence
  - KL: regularize latent to prior N(0,1)
- **Applications**
  - Generation, representation learning

### 2.6.3 Generative Adversarial Networks (GAN)
- **Two Networks**
  - Generator: noise → fake sample
  - Discriminator: real vs fake classifier
- **Minimax Game**
  - min_G max_D V(D,G)
- **Training Challenges**
  - Mode collapse
  - Vanishing gradients
  - Non-convergence
- **GAN Variants**
  - DCGAN: convolutional architecture
  - WGAN: Wasserstein distance
  - CGAN: conditional generation
  - StyleGAN: style-based generator

### 2.6.4 Diffusion Models
- **Forward Process**
  - Gradually add Gaussian noise
  - q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_tI)
- **Reverse Process**
  - Learn to denoise
  - p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
- **Training Objective**
  - Predict noise (simplified)
  - L = E[||ε - ε_θ(x_t, t)||²]
- **Sampling**
  - Start from random noise
  - Iteratively denoise
- **DDPM, DDIM, Stable Diffusion**
  - Latent diffusion for efficiency
  - Text conditioning via cross-attention

### 🔴 Commonly Missed / Underrated Topics (Deep Learning)

- **Numerical Stability**
  - Log-sum-exp trick
  - Softmax numerical issues
  - Gradient clipping strategies
- **Dead ReLU and Activation Analysis**
  - Monitoring activation statistics
  - Dead neuron detection and revival
- **Batch Size Effects**
  - Generalization gap (large batch → poor generalization)
  - Linear scaling rule for learning rate
  - Critical batch size
- **Sharpness-Aware Minimization (SAM)**
  - Optimize for flat minima
  - Improved generalization
- **Neural Tangent Kernel (NTK)**
  - Infinite width limit behavior
  - Connection to kernel methods
- **Lottery Ticket Hypothesis**
  - Sparse subnetworks train well
  - Finding winning tickets
- **Knowledge Distillation**
  - Teacher-student training
  - Soft targets with temperature
  - Applications: model compression
- **Neural Architecture Search (NAS)**
  - Automated architecture design
  - Efficiency vs performance tradeoff
- **Test-Time Adaptation**
  - Adapt model at inference
  - Handle distribution shift
- **Uncertainty Quantification**
  - Aleatoric (data) vs epistemic (model) uncertainty
  - MC Dropout, Deep Ensembles
  - Evidential deep learning

### 🎯 Interview Focus (Deep Learning)

**Beginner Level:**
- Why ReLU over sigmoid?
- Explain backpropagation step-by-step
- What is batch normalization and why use it?
- Vanishing gradients: causes and solutions
- CNN vs RNN: when to use which

**Intermediate Level:**
- Derive backprop for a simple network
- Compare LSTM vs GRU tradeoffs
- Why scaled dot-product attention? (Answer: √d_k prevents softmax saturation)
- Explain VAE reparameterization trick
- GAN training instability causes

**Senior Level:**
- Design a custom architecture for [specific problem]
- Optimize training for 100B parameter model
- Implement efficient attention for long sequences
- Debug why validation loss increases while training decreases
- Design distributed training strategy

**Differentiating Questions:**
- "Why might batch norm hurt RNNs?" (Answer: batch statistics vary with sequence length)
- "Your GAN generates only one type of output — what's happening?" (Answer: mode collapse)
- "Design a neural network for a resource-constrained edge device" (Answer: knowledge distillation, quantization, pruning)

---

# 3. Statistics & Mathematics

## 3.1 Probability Theory

### 3.1.1 Fundamentals
- **Sample Space and Events**
  - Ω: set of all outcomes
  - Events: subsets of Ω
- **Probability Axioms**
  - P(A) ≥ 0
  - P(Ω) = 1
  - P(A ∪ B) = P(A) + P(B) for disjoint events
- **Conditional Probability**
  - P(A|B) = P(A ∩ B) / P(B)
  - Chain rule: P(A,B,C) = P(A)P(B|A)P(C|A,B)
- **Independence**
  - P(A ∩ B) = P(A)P(B)
  - Conditional independence

### 3.1.2 Bayes Theorem
- **Formula**
  - P(A|B) = P(B|A)P(A) / P(B)
- **Components**
  - Prior P(A): initial belief
  - Likelihood P(B|A): evidence given hypothesis
  - Posterior P(A|B): updated belief
  - Evidence P(B): normalizing constant
- **Applications**
  - Bayesian inference
  - Naive Bayes classifier
  - Bayesian optimization

### 3.1.3 Random Variables
- **Discrete Random Variables**
  - PMF: P(X = x)
  - Support: possible values
  - Bernoulli, Binomial, Poisson, Geometric
- **Continuous Random Variables**
  - PDF: f(x), P(a ≤ X ≤ b) = ∫f(x)dx
  - CDF: F(x) = P(X ≤ x)
  - Uniform, Normal, Exponential, Gamma, Beta
- **Expectation**
  - E[X] = Σx·P(X=x) or ∫x·f(x)dx
  - Linearity: E[aX + bY] = aE[X] + bE[Y]
  - Law of total expectation
- **Variance**
  - Var(X) = E[(X - μ)²] = E[X²] - E[X]²
  - Var(aX + b) = a²Var(X)
  - Covariance and correlation

### 3.1.4 Distributions
- **Normal/Gaussian Distribution**
  - N(μ, σ²): PDF = (1/√(2πσ²))exp(-(x-μ)²/(2σ²))
  - 68-95-99.7 rule
  - Central Limit Theorem connection
- **Multivariate Normal**
  - Mean vector μ, covariance matrix Σ
  - Mahalanobis distance
- **Bernoulli and Binomial**
  - Bernoulli: single trial, p success
  - Binomial: n independent Bernoulli trials
- **Poisson**
  - Count events in fixed interval
  - P(X=k) = (λ^k·e^{-λ})/k!
- **Exponential**
  - Time between Poisson events
  - Memoryless property
- **Beta and Dirichlet**
  - Conjugate priors for Binomial/Multinomial
  - Beta: distribution over probabilities

### 3.1.5 Limit Theorems
- **Law of Large Numbers**
  - Sample mean converges to expected value
  - Weak vs strong LLN
- **Central Limit Theorem**
  - Sum of i.i.d. variables → Normal
  - Enables confidence intervals
  - Justifies normality assumptions

## 3.2 Statistical Inference

### 3.2.1 Estimation
- **Point Estimation**
  - Method of moments
  - Maximum Likelihood Estimation (MLE)
    - Find θ maximizing likelihood L(θ|data)
    - Log-likelihood for computational ease
    - Properties: consistency, asymptotic normality
  - Maximum A Posteriori (MAP)
    - MLE + prior: argmax P(θ|data)
    - Regularization interpretation
- **Interval Estimation**
  - Confidence intervals
  - Interpretation: 95% of intervals contain true parameter
  - Bootstrap confidence intervals

### 3.2.2 Hypothesis Testing
- **Framework**
  - Null hypothesis H₀ vs Alternative H₁
  - Type I error (false positive): reject true H₀
  - Type II error (false negative): fail to reject false H₀
  - Significance level α (typically 0.05)
  - Power: 1 - P(Type II error)
- **Test Statistics**
  - Z-test: known variance, large samples
  - T-test: unknown variance
  - Chi-squared: categorical data
  - F-test: comparing variances
- **P-value**
  - Probability of observing data given H₀
  - Not P(H₀|data) — common misinterpretation
- **Common Tests**
  - One-sample t-test
  - Two-sample t-test (independent/paired)
  - ANOVA: compare multiple means
  - Chi-squared test of independence
  - Kolmogorov-Smirnov test

### 3.2.3 Bayesian Statistics
- **Prior Selection**
  - Informative vs uninformative
  - Conjugate priors
  - Jeffreys prior
- **Posterior Computation**
  - Analytical solutions (conjugate cases)
  - MCMC: Metropolis-Hastings, Gibbs sampling
  - Variational inference
- **Credible Intervals**
  - Bayesian counterpart to confidence intervals
  - Direct probability interpretation

## 3.3 Linear Algebra

### 3.3.1 Vectors and Matrices
- **Vector Operations**
  - Addition, scalar multiplication
  - Dot product: a·b = Σaᵢbᵢ = |a||b|cos(θ)
  - Norms: L1, L2 (Euclidean), L∞
  - Orthogonality: a·b = 0
- **Matrix Operations**
  - Addition, multiplication
  - Transpose, inverse
  - Trace, determinant
  - Rank: dimension of column/row space

### 3.3.2 Matrix Properties
- **Special Matrices**
  - Identity matrix I
  - Diagonal matrix
  - Symmetric matrix: A = Aᵀ
  - Orthogonal matrix: QᵀQ = I
  - Positive definite: xᵀAx > 0 for all x ≠ 0
- **Eigenvalues and Eigenvectors**
  - Av = λv
  - Characteristic polynomial
  - Spectral decomposition
  - Applications: PCA, PageRank
- **Singular Value Decomposition (SVD)**
  - A = UΣVᵀ
  - U: left singular vectors (eigenvectors of AAᵀ)
  - Σ: singular values (sqrt of eigenvalues)
  - V: right singular vectors (eigenvectors of AᵀA)
  - Applications: dimensionality reduction, pseudoinverse

### 3.3.3 Linear Systems
- **Solving Ax = b**
  - Gaussian elimination
  - LU decomposition
  - Matrix inversion (avoid for large systems)
- **Least Squares**
  - Minimize ||Ax - b||²
  - Normal equations: AᵀAx = Aᵀb
  - Pseudoinverse: x = A⁺b

## 3.4 Calculus & Optimization

### 3.4.1 Differential Calculus
- **Derivatives**
  - Limit definition
  - Common derivatives: power, exponential, log, trig
  - Partial derivatives for multivariate functions
- **Gradient**
  - ∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]
  - Direction of steepest ascent
- **Chain Rule**
  - ∂f/∂x = ∂f/∂u · ∂u/∂x
  - Foundation of backpropagation
- **Jacobian and Hessian**
  - Jacobian: matrix of first derivatives
  - Hessian: matrix of second derivatives
  - Positive definite Hessian → local minimum

### 3.4.2 Optimization Theory
- **Convexity**
  - Convex set: line segment between any two points in set
  - Convex function: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
  - Local minimum = global minimum for convex functions
- **Lagrange Multipliers**
  - Constrained optimization
  - L(x, λ) = f(x) + λg(x)
- **Karush-Kuhn-Tucker (KKT) Conditions**
  - Generalization for inequality constraints
  - Necessary conditions for optimality

### 3.4.3 Information Theory
- **Entropy**
  - H(X) = -ΣP(x)logP(x)
  - Uncertainty in random variable
  - Maximum for uniform distribution
- **Cross-Entropy**
  - H(p,q) = -Σp(x)logq(x)
  - Loss function for classification
- **KL Divergence**
  - D_KL(p||q) = Σp(x)log(p(x)/q(x))
  - Measure of distribution difference
  - Not symmetric
- **Mutual Information**
  - I(X;Y) = H(X) - H(X|Y)
  - Measures dependence between variables

### 🔴 Commonly Missed / Underrated Topics (Statistics & Math)

- **The Bootstrap**
  - Resampling for uncertainty estimation
  - Bias-corrected bootstrap
  - Computational vs theoretical approaches
- **Multiple Testing Correction**
  - Family-wise error rate (Bonferroni)
  - False Discovery Rate (Benjamini-Hochberg)
  - Importance in feature selection
- **Simpson's Paradox**
  - Trend reverses when groups combined
  - Confounding variable explanation
  - Real-world examples in ML
- **Survivorship Bias**
  - Only analyzing "survivors"
  - Common in financial ML
- **Correlation vs Causation**
  - Confounding variables
  - Instrumental variables
  - Natural experiments
- **Concentration Inequalities**
  - Markov, Chebyshev, Chernoff bounds
  - Hoeffding's inequality
  - PAC learning theory
- **Matrix Calculus**
  - Gradients of matrix expressions
  - Common derivatives in ML
- **Fisher Information**
  - Curvature of log-likelihood
  - Cramer-Rao lower bound
- **Exchangeability**
  - Order doesn't matter
  - De Finetti's theorem
  - Bayesian nonparametrics

### 🎯 Interview Focus (Statistics & Math)

**Beginner Level:**
- Explain p-value correctly
- Bayes theorem application
- MLE vs MAP difference
- Eigenvalues and eigenvectors intuition
- Why is gradient the direction of steepest ascent?

**Intermediate Level:**
- Derive MLE for Gaussian mean and variance
- Explain confidence interval vs credible interval
- When does CLT apply and when not?
- SVD and its applications
- KL divergence properties and interpretation

**Senior Level:**
- Design statistical test for A/B testing
- Handle multiple comparisons in feature selection
- Derive gradient for custom loss function
- Analyze convergence of optimization algorithm
- Design Bayesian model for uncertainty quantification

**Differentiating Questions:**
- "Your p-value is 0.04, what does this mean?" (Answer: if H₀ true, 4% chance of this extreme data — NOT 96% probability H₁ is true)
- "Why might correlation not imply predictive power?" (Answer: non-linear relationships, outliers, range restriction)
- "Explain why cross-entropy is the right loss for classification" (Answer: maximum likelihood for Bernoulli/multinomial)

---

# 4. Natural Language Processing (NLP)

## 4.1 Text Preprocessing

### 4.1.1 Tokenization
- **Word Tokenization**
  - Space-based (English)
  - Rule-based approaches
  - Handling punctuation
- **Subword Tokenization**
  - Byte Pair Encoding (BPE)
    - Merge frequent character pairs
    - Used in GPT, RoBERTa
  - WordPiece
    - Similar to BPE, used in BERT
  - SentencePiece
    - Language-agnostic, used in T5, XLNet
  - Unigram Language Model
    - Used in XLNet
- **Character Tokenization**
  - No OOV issues
  - Longer sequences
- **Challenges**
  - Out-of-vocabulary (OOV) words
  - Languages without word boundaries
  - URLs, emails, special tokens

### 4.1.2 Normalization
- **Lowercasing**
  - Tradeoff: lose case information
- **Punctuation Removal**
  - Context-dependent necessity
- **Stop Word Removal**
  - Common words with little meaning
  - Modern trend: keep for context
- **Stemming**
  - Porter, Snowball stemmers
  - Crude rule-based reduction
- **Lemmatization**
  - Dictionary-based to root form
  - More accurate than stemming
- **Spell Correction**
  - Edit distance based
  - Context-aware correction

### 4.1.3 Text Representations
- **One-Hot Encoding**
  - Binary vector per word
  - High dimensionality, no semantics
- **Bag of Words (BoW)**
  - Word frequency vector
  - Loses word order
- **TF-IDF**
  - TF: term frequency in document
  - IDF: inverse document frequency
  - Weights: TF × IDF
  - Downweights common words
- **N-grams**
  - Capture local word order
  - Unigram, bigram, trigram
  - Sparse representation

## 4.2 Word Embeddings

### 4.2.1 Classical Embeddings
- **Word2Vec**
  - **CBOW**: predict word from context
  - **Skip-gram**: predict context from word
  - Negative sampling for efficiency
  - Subsampling frequent words
  - Window size hyperparameter
- **GloVe (Global Vectors)**
  - Factorize word-word co-occurrence matrix
  - Combines global statistics + local context
- **FastText**
  - Subword information
  - Handles OOV via character n-grams
  - Better for morphologically rich languages

### 4.2.2 Embedding Properties
- **Semantic Relationships**
  - king - man + woman ≈ queen
  - Analogies in vector space
- **Limitations**
  - Static embeddings: one vector per word
  - Polysemy (multiple meanings)
  - Context-independent

### 4.2.3 Contextualized Embeddings
- **ELMo (Embeddings from Language Models)**
  - BiLSTM-based
  - Deep contextualized representations
  - Character-based, handles OOV
- **Contextual Word Vectors**
  - Different representation per context
  - Solves polysemy problem

## 4.3 Sequence Modeling

### 4.3.1 Traditional NLP Models
- **Hidden Markov Models (HMM)**
  - For POS tagging, NER
  - Emission and transition probabilities
- **Conditional Random Fields (CRF)**
  - Discriminative sequence model
  - Better than HMM for labeling
- **Maximum Entropy Markov Models**
  - Logistic regression for sequences

### 4.3.2 Neural NLP Architectures
- **CNN for Text**
  - n-gram feature detectors
  - Parallel processing
  - Text classification applications
- **RNN for Text**
  - Sequential processing
  - LSTM/GRU for long dependencies
  - Encoder-decoder for translation
- **Bidirectional LSTM**
  - Context from both directions
  - Standard before Transformers

## 4.4 NLP Tasks

### 4.4.1 Text Classification
- **Sentiment Analysis**
  - Binary, multi-class, fine-grained
  - Aspect-based sentiment
- **Topic Classification**
  - News categorization
  - Intent classification
- **Spam Detection**
  - Feature engineering approaches
  - Deep learning approaches

### 4.4.2 Sequence Labeling
- **Part-of-Speech (POS) Tagging**
  - Assign grammatical categories
  - Ambiguity resolution
- **Named Entity Recognition (NER)**
  - Identify entities: person, org, location, etc.
  - BIO tagging scheme
  - Nested NER challenges
- **Chunking**
  - Group words into phrases

### 4.4.3 Parsing
- **Syntactic Parsing**
  - Constituency parsing (phrase structure)
  - Dependency parsing (word relationships)
- **Semantic Parsing**
  - Meaning representation
  - SQL generation from natural language

### 4.4.4 Text Generation
- **Language Modeling**
  - Predict next word given context
  - Perplexity evaluation
- **Machine Translation**
  - Statistical MT (older)
  - Neural MT (current)
  - Attention mechanism importance
  - BLEU score evaluation
- **Summarization**
  - Extractive: select sentences
  - Abstractive: generate new text
  - ROUGE score evaluation
- **Question Answering**
  - Reading comprehension
  - Open-domain QA
  - SQuAD, Natural Questions datasets

### 4.4.5 Advanced NLP Tasks
- **Coreference Resolution**
  - Link mentions to same entity
  - "He" → "John"
- **Relation Extraction**
  - Identify relationships between entities
- **Textual Entailment**
  - Natural Language Inference (NLI)
  - Entailment, contradiction, neutral
- **Semantic Textual Similarity**
  - STS benchmarks
  - Sentence embedding approaches

## 4.5 Modern NLP (Pre-Transformer Era)

### 4.5.1 Attention in NLP
- **Seq2Seq with Attention**
  - Bahdanau attention
  - Luong attention
  - Alignment visualization
- **Self-Attention Emergence**
  - Intra-sentence attention
  - Long-range dependency handling

### 4.5.2 Transfer Learning in NLP
- **ULMFiT**
  - Universal Language Model Fine-tuning
  - Discriminative fine-tuning
  - Gradual unfreezing
- **Pre-training + Fine-tuning Paradigm**
  - General pre-training on large corpus
  - Task-specific fine-tuning

### 🔴 Commonly Missed / Underrated Topics (NLP)

- **Tokenization Pitfalls**
  - BPE merge order effects
  - Token boundaries and model behavior
  - Adversarial tokenization attacks
- **Handling Long Documents**
  - Hierarchical attention
  - Sliding window approaches
  - Longformer, BigBird sparse attention
- **Low-Resource Languages**
  - Cross-lingual transfer
  - Multilingual embeddings
  - Data augmentation strategies
- **Evaluation Challenges**
  - BLEU limitations (n-gram matching)
  - Human evaluation necessity
  - Diversity vs quality tradeoff
- **Bias in NLP**
  - Gender bias in word embeddings
  - Stereotypical associations
  - Debiasing techniques
- **Adversarial Examples in NLP**
  - Character-level perturbations
  - Paraphrase attacks
  - Robustness evaluation
- **Interpretability in NLP**
  - Attention visualization (not always explanation)
  - LIME/SHAP for text
  - Probing classifiers
- **Data Leakage in NLP**
  - Pre-training data contamination
  - Test set in pre-training corpus
  - Deduplication importance
- **Efficient NLP**
  - Knowledge distillation for BERT (DistilBERT)
  - Quantization of language models
  - Pruning techniques

### 🎯 Interview Focus (NLP)

**Beginner Level:**
- Explain Word2Vec CBOW vs Skip-gram
- TF-IDF calculation and intuition
- Tokenization challenges
- Why RNNs struggle with long sequences

**Intermediate Level:**
- How does BPE tokenization work?
- Explain attention mechanism in seq2seq
- NER tagging schemes (BIO)
- Machine translation evaluation (BLEU limitations)

**Senior Level:**
- Design custom tokenizer for domain-specific text
- Handle 10K+ token documents efficiently
- Address bias in word embeddings
- Design multilingual NLP pipeline

**Differentiating Questions:**
- "Why might attention weights not be good explanations?" (Answer: attention can be uniform but model still relies on specific features)
- "Your NER model fails on nested entities — how to fix?" (Answer: BIOES schemes, span-based approaches)
- "Why did Transformers replace RNNs for NLP?" (Answer: parallelization, long-range dependencies, no vanishing gradients)

---

# 5. Large Language Models (LLMs)

## 5.1 Transformer Architecture

### 5.1.1 Core Components
- **Input Embeddings**
  - Token embeddings + positional embeddings
  - Learned vs sinusoidal positional encoding
  - Rotary Position Embedding (RoPE)
  - ALiBi (Attention with Linear Biases)
- **Multi-Head Self-Attention**
  - Q, K, V projections
  - Scaled dot-product: Attention(Q,K,V) = softmax(QKᵀ/√d_k)V
  - Multiple heads for different representation subspaces
  - Masking for autoregressive generation
- **Feed-Forward Networks**
  - Position-wise FFN
  - GELU activation (common in modern LLMs)
  - Expansion factor (typically 4×)
- **Layer Normalization**
  - Pre-norm vs post-norm
  - Pre-norm: more stable training
- **Residual Connections**
  - Enable gradient flow in deep networks
  - Pre-norm: LayerNorm → Sub-layer → Add

### 5.1.2 Encoder-Decoder vs Decoder-Only
- **Encoder-Decoder (T5, BART)**
  - Encoder: bidirectional attention
  - Decoder: causal (autoregressive) attention
  - Cross-attention between encoder and decoder
  - Good for translation, summarization
- **Decoder-Only (GPT, LLaMA)**
  - Causal attention throughout
  - Autoregressive generation
  - Simpler, scalable
  - Current dominant architecture
- **Encoder-Only (BERT)**
  - Bidirectional attention
  - Not for generation
  - Good for understanding tasks

### 5.1.3 Architectural Variants
- **Parallel Attention (PaLM)**
  - Attention + FFN in parallel
  - Faster training
- **Multi-Query Attention**
  - Shared K, V across heads
  - Faster inference, slight quality drop
- **Grouped Query Attention (GQA)**
  - Middle ground between MHA and MQA
  - Balance speed and quality
- **Sliding Window Attention**
  - Local attention pattern
  - Handle longer sequences
- **Sparse Attention Patterns**
  - Longformer: global + local + dilated
  - BigBird: random + window + global
  - Linear attention complexity

## 5.2 Training LLMs

### 5.2.1 Pre-training
- **Objective Functions**
  - Causal Language Modeling (CLM): predict next token
  - Masked Language Modeling (MLM): predict masked tokens
  - Prefix LM: predict continuation
  - Span corruption (T5-style)
- **Training Data**
  - Web crawl (Common Crawl, C4)
  - Books, Wikipedia, code
  - Data quality filtering
  - Deduplication importance
- **Scale Laws**
  - Kaplan scaling laws
  - Loss ∝ N^(-α), N^(-β) for model size and data
  - Chinchilla scaling: optimal compute allocation
  - Over-training trend (Llama 3)
- **Training Infrastructure**
  - Data parallelism
  - Model parallelism (tensor, pipeline)
  - ZeRO optimizer states sharding
  - Mixed precision training (FP16/BF16)
  - Gradient accumulation

### 5.2.2 Tokenization for LLMs
- **BPE in Practice**
  - Vocabulary size tradeoff (32K-100K+ typical)
  - Pre-tokenizer: splitting before BPE
  - Special tokens: <s>, </s>, <pad>, <unk>
- **Byte-Level BPE**
  - GPT-2, RoBERTa
  - No unknown tokens
- **SentencePiece**
  - Language-agnostic
  - Directly on raw text
- **Impact on Performance**
  - Tokenization affects effective context length
  - Multi-language tokenization challenges

### 5.2.3 Optimization for LLMs
- **Learning Rate Schedules**
  - Warmup + cosine decay
  - Warmup prevents early instability
- **Gradient Clipping**
  - Prevent exploding gradients
- **Optimizer Choice**
  - AdamW standard
  - Lion optimizer (newer)
- **Batch Size Scaling**
  - Large batches with LR scaling
  - Gradient accumulation for memory

## 5.3 Fine-tuning LLMs

### 5.3.1 Full Fine-tuning
- **Update All Parameters**
  - Computationally expensive
  - Risk of catastrophic forgetting
  - Requires significant GPU memory
- **Learning Rate Considerations**
  - Lower LR than pre-training
  - Layer-wise learning rates

### 5.3.2 Parameter-Efficient Fine-Tuning (PEFT)
- **LoRA (Low-Rank Adaptation)**
  - W = W₀ + BA, where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)
  - Train low-rank matrices instead of full weights
  - Reduces trainable parameters by ~10,000×
  - Mergeable for inference
  - Rank selection tradeoff
- **QLoRA**
  - Quantized base model (4-bit)
  - LoRA on top
  - Enables fine-tuning on consumer GPUs
- **Adapter Layers**
  - Small bottleneck layers after attention/FFN
  - Original weights frozen
  - Task-specific adapters
- **Prefix Tuning / P-Tuning**
  - Train continuous prompts
  - Virtual tokens prepended
- **Prompt Tuning**
  - Soft prompts optimized per task
  - Scale with model size
- **IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)**
  - Learn scaling vectors
  - More expressive than LoRA in some cases

### 5.3.3 Instruction Tuning
- **Supervised Fine-Tuning (SFT)**
  - Train on (instruction, response) pairs
  - Format: prompts with special tokens
- **Dataset Construction**
  - Quality over quantity
  - Diversity of tasks
  - Human-written vs synthetic data
- **Multi-turn Conversations**
  - Chat format training
  - Role tokens: user, assistant, system

## 5.4 Alignment

### 5.4.1 Reinforcement Learning from Human Feedback (RLHF)
- **Three-Stage Process**
  1. SFT: supervised fine-tuning
  2. Reward Model: train from preferences
  3. RL: optimize policy with PPO
- **Reward Model Training**
  - Data: comparisons (A > B)
  - Bradley-Terry model
  - Loss: -log σ(r_θ(x, y_w) - r_θ(x, y_l))
- **PPO (Proximal Policy Optimization)**
  - Clipped surrogate objective
  - KL penalty to stay close to reference
  - Reward hacking challenges
- **Challenges**
  - Reward model overoptimization
  - Distribution shift
  - Alignment tax (capability reduction)

### 5.4.2 Alternative Alignment Methods
- **Direct Preference Optimization (DPO)**
  - Skip explicit reward model
  - Directly optimize from preferences
  - Simpler, often as effective
- **Kahneman-Tversky Optimization (KTO)**
  - Binary feedback (good/bad)
  - No pairwise comparisons needed
- **Constitutional AI**
  - Self-improvement via principles
  - RL from AI feedback (RLAIF)
- **Rejection Sampling Fine-Tuning (RFT)**
  - Generate multiple responses
  - Fine-tune on best ones

### 5.4.3 Safety and Harmlessness
- **Red Teaming**
  - Adversarial testing
  - Uncover failure modes
- **Content Moderation**
  - Input/output filtering
  - Moderation classifiers
- **Refusal Training**
  - Train model to decline harmful requests
  - Balanced refusal (not over-refusing)

## 5.5 LLM Inference

### 5.5.1 Decoding Strategies
- **Greedy Decoding**
  - Always pick highest probability token
  - Deterministic, often suboptimal
- **Beam Search**
  - Keep top-k hypotheses
  - Better for short sequences
  - Repetition issues for long text
- **Sampling Methods**
  - Temperature: scale logits before softmax
    - T < 1: more focused, conservative
    - T > 1: more random, creative
  - Top-k: sample from k most likely
  - Top-p (nucleus): sample from smallest set with cumulative prob ≥ p
  - Typical sampling: based on expected information content
- **Repetition Penalty**
  - Down-weight previously generated tokens
- **Contrastive Search**
  - Balance model confidence with degeneration penalty

### 5.5.2 Efficient Inference
- **KV Cache**
  - Store key-value pairs from previous tokens
  - Avoid recomputation
  - Memory grows with sequence length
- **Quantization**
  - INT8, INT4 weight quantization
  - GPTQ: post-training quantization
  - AWQ: activation-aware quantization
- **Speculative Decoding**
  - Draft model predicts multiple tokens
  - Target model verifies in parallel
  - Speedup without quality loss
- **Continuous Batching**
  - Dynamic batching during generation
  - Higher throughput
- **PagedAttention (vLLM)**
  - Efficient KV cache memory management
  - Paging-like allocation

### 5.5.3 Context Length Extension
- **Position Interpolation**
  - Scale position embeddings
  - Fine-tune for longer context
- **NTK-Aware Scaling**
  - Non-linear position interpolation
  - Better extrapolation
- **YaRN, LongRoPE**
  - Advanced position encoding methods
- **Ring Attention**
  - Distributed attention computation
  - Arbitrarily long sequences

## 5.6 LLM Capabilities and Evaluation

### 5.6.1 Emergent Abilities
- **In-Context Learning**
  - Few-shot prompting
  - Zero-shot generalization
  - Task adaptation without parameter updates
- **Chain-of-Thought (CoT)**
  - Step-by-step reasoning
  - "Let's think step by step"
  - Self-consistency decoding
- **Advanced Prompting**
  - Tree of Thoughts
  - Graph of Thoughts
  - ReAct (Reasoning + Acting)
  - Reflexion (self-improvement)

### 5.6.2 Benchmarks
- **Knowledge**
  - MMLU: multi-task understanding
  - TriviaQA, Natural Questions
- **Reasoning**
  - GSM8K: math word problems
  - HumanEval: code generation
  - BBH: big bench hard
- **Instruction Following**
  - MT-bench, AlpacaEval
  - Human preference evaluation
- **Safety**
  - TruthfulQA
  - Harmful request testing

### 5.6.3 LLM Limitations
- **Hallucinations**
  - Confident false statements
  - Factual vs contextual hallucinations
  - Mitigation: RAG, fact-checking
- **Knowledge Cutoff**
  - No information after training date
  - Solutions: RAG, tool use
- **Reasoning Limitations**
  - Struggles with complex multi-step logic
  - Arithmetic errors
- **Context Window**
  - Limited attention span
  - Lost in the middle problem
- **Calibration**
  - Overconfident predictions
  - Poor probability calibration

### 🔴 Commonly Missed / Underrated Topics (LLMs)

- **Tokenization Edge Cases**
  - "SolidGoldMagikarp" glitch token phenomenon
  - Token boundary attacks
  - Reversible tokenization
- **Context Window Limitations**
  - "Lost in the middle" — poor retrieval from middle of long context
  - Position bias in attention
  - Solutions: LongLoRA, prompt reordering
- **Data Contamination**
  - Test sets in pre-training data
  - Benchmark validity concerns
  - Deduplication strategies
- **Emergence and Scaling**
  - Sharp capability jumps
  - Unpredictable from smaller models
  - Whether emergence is metric-dependent
- **Reward Hacking in RLHF**
  - Exploiting reward model weaknesses
  - Overoptimization detection
  - KL divergence monitoring
- **Inference-Time Compute Scaling**
  - Test-time computation tradeoffs
  - Best-of-N sampling
  - Process reward models
- **Mixture of Experts (MoE)**
  - Sparse activation
  - Conditional computation
  - Load balancing challenges
  - Switch Transformer, Mixtral
- **Multi-Modal LLMs**
  - Vision-language models
  - CLIP, LLaVA, GPT-4V
  - Alignment of modalities
- **Tool Use and Function Calling**
  - Teaching LLMs to use external tools
  - API calling formats
  - ReAct pattern implementation
- **Model Merging**
  - Task Arithmetic
  - SLERP, TIES, DARE
  - Creating ensembles without inference cost

### 🎯 Interview Focus (LLMs)

**Beginner Level:**
- Explain Transformer self-attention mechanism
- Difference between encoder and decoder attention
- What is LoRA and why use it?
- Temperature and top-p sampling

**Intermediate Level:**
- Why scale by √d_k in attention?
- Explain RLHF pipeline and challenges
- Compare DPO vs PPO for alignment
- KV cache: what, why, and memory implications
- Handle long context beyond training length

**Senior Level:**
- Design distributed training for 100B model
- Optimize inference latency for production
- Address reward hacking in RLHF
- Design evaluation suite for domain-specific LLM
- Implement speculative decoding

**Differentiating Questions:**
- "Your model hallucinates on factual questions — what's your debugging approach?" (Answer: RAG integration, uncertainty quantification, retrieval-augmented generation)
- "Why might beam search produce worse results than sampling for creative tasks?" (Answer: mode collapse, repetitive outputs, lack of diversity)
- "Design a system to detect when LLM is uncertain" (Answer: token probability entropy, calibration metrics, abstention training)

---

# 6. Retrieval-Augmented Generation (RAG)

## 6.1 RAG Fundamentals

### 6.1.1 Core Concept
- **Motivation**
  - Ground LLM in external knowledge
  - Reduce hallucinations
  - Access proprietary/private data
  - Handle knowledge cutoff
- **Basic Pipeline**
  1. User query → 2. Retrieve documents → 3. Augment prompt → 4. Generate response
- **Benefits**
  - Citation/attribution
  - Dynamic knowledge updates
  - Domain adaptation without fine-tuning

### 6.1.2 RAG vs Fine-tuning
- **When to Use RAG**
  - Frequently changing knowledge
  - Need for attribution
  - Limited training data
  - Multiple domains
- **When to Use Fine-tuning**
  - Style/tone adaptation
  - Specific task optimization
  - Teaching new patterns
- **Hybrid Approaches**
  - RAG + fine-tuned retriever
  - Fine-tuned generator with RAG

## 6.2 Document Processing

### 6.2.1 Chunking Strategies
- **Fixed-Size Chunking**
  - Simple, predictable
  - May split semantic units
- **Semantic Chunking**
  - Preserve sentence/paragraph boundaries
  - Natural language boundaries
- **Recursive Chunking**
  - Hierarchical: paragraphs → sentences
  - Balance size and coherence
- **Agentic Chunking**
  - LLM-based document segmentation
  - Content-aware splitting
- **Chunk Size Tradeoffs**
  - Small chunks: precise retrieval, more noise
  - Large chunks: more context, dilution
  - Typical: 200-500 tokens
- **Chunk Overlap**
  - Preserve context across boundaries
  - Typically 10-20% overlap

### 6.2.2 Document Enhancement
- **Metadata Extraction**
  - Document source, date, author
  - Section headers
  - Hierarchical structure
- **Summarization**
  - Hierarchical summaries
  - Parent document mapping
- **Entity Extraction**
  - Named entities for filtering
  - Structured metadata

## 6.3 Embedding Models

### 6.3.1 Dense Retrieval
- **Bi-Encoder Architecture**
  - Separate encoding of query and document
  - Dot product/cosine similarity
  - Efficient at scale
- **Popular Models**
  - Sentence-BERT (SBERT)
  - E5 (EmbEddings from bidirectional Encoder)
  - BGE (BAAI General Embedding)
  - GTE (General Text Embeddings)
  - OpenAI embeddings
- **Training Approaches**
  - Contrastive learning
  - In-batch negatives
  - Hard negative mining
  - InfoNCE loss

### 6.3.2 Late Interaction Models
- **ColBERT**
  - Token-level interactions
  - MaxSim operator
  - More accurate, slower than bi-encoder
- **ColBERTv2**
  - Compression for efficiency
  - PLAID indexing

### 6.3.3 Sparse Retrieval
- **BM25**
  - Probabilistic retrieval function
  - Term frequency + inverse document frequency
  - Length normalization
- **SPLADE**
  - Learned sparse representations
  - Neural expansion
  - Best of sparse and dense

### 6.3.4 Hybrid Retrieval
- **Combining Sparse + Dense**
  - Linear combination of scores
  - Reciprocal Rank Fusion (RRF)
  - Learned fusion
- **When Each Excels**
  - Sparse: exact matches, rare terms
  - Dense: semantic similarity, paraphrases

## 6.4 Vector Databases

### 6.4.1 Vector Indexing
- **Flat Index (Brute Force)**
  - Exact search, slow at scale
  - Small datasets (<10K)
- **Approximate Nearest Neighbor (ANN)**
  - Trade accuracy for speed
  - **HNSW (Hierarchical Navigable Small World)**
    - Graph-based index
    - Multi-layer structure
    - Fast search with high recall
  - **IVF (Inverted File Index)**
    - Cluster then search
    - Coarse quantizer
  - **PQ (Product Quantization)**
    - Compress vectors
    - Memory efficient
  - **ScaNN**
    - Google's optimized ANN
    - Asymmetric hashing

### 6.4.2 Vector Database Options
- **Pinecone**
  - Managed, serverless
  - Metadata filtering
- **Weaviate**
  - GraphQL interface
  - Modular ML integrations
- **Milvus/Zilliz**
  - GPU acceleration
  - Distributed architecture
- **Chroma**
  - Developer-friendly
  - Local-first
- **pgvector**
  - PostgreSQL extension
  - ACID compliance
- **FAISS**
  - Meta's library
  - Highly optimized
  - In-memory

### 6.4.3 Production Considerations
- **Sharding and Replication**
  - Horizontal scaling
  - Fault tolerance
- **Metadata Filtering**
  - Pre-filter vs post-filter
  - Hybrid search with constraints
- **Multi-tenancy**
  - Namespace isolation
  - Access control

## 6.5 Retrieval Strategies

### 6.5.1 Query Processing
- **Query Expansion**
  - Pseudo-relevance feedback
  - LLM-based expansion
  - HyDE (Hypothetical Document Embeddings)
- **Query Rewriting**
  - Multi-query expansion
  - Step-back prompting
  - Query decomposition
- **Query Embedding**
  - Same model as documents
  - Task-specific fine-tuning

### 6.5.2 Retrieval Methods
- **Single-Stage Retrieval**
  - Direct top-k from index
  - Simple and fast
- **Multi-Stage Retrieval**
  - Coarse retrieval → Reranking
  - Balance efficiency and accuracy
- **Iterative Retrieval**
  - Retrieve → Generate → Retrieve more
  - IRCoT, Self-RAG

### 6.5.3 Reranking
- **Cross-Encoder Reranker**
  - Joint encoding of query + document
  - More accurate than bi-encoder
  - Slower, use on top-k results
- **LLM Reranking**
  - Pointwise, pairwise, listwise approaches
  - Zero-shot reranking
- **Cohere Rerank, BGE Reranker**
  - Specialized reranking models

## 6.6 Advanced RAG Patterns

### 6.6.1 Pre-Retrieval Enhancement
- **Document Indexing**
  - Parent-document retrieval
  - Hierarchical indexing
  - Summary-based indexing
- **Hypothetical Questions**
  - Generate questions documents answer
  - Index questions, retrieve documents

### 6.6.2 Post-Retrieval Enhancement
- **Context Compression**
  - Relevant segment extraction
  - LLM-based compression
- **Repacking**
  - Order documents for best generation
  - Lost in the middle mitigation
- **Prompt Engineering**
  - System prompts for RAG
  - Citation requirements

### 6.6.3 Agentic RAG
- **Self-Reflection**
  - Evaluate retrieved context sufficiency
  - Iterative refinement
- **Self-RAG**
  - Retrieve on demand
  - Critic tokens for quality
- **Corrective RAG (CRAG)**
  - Grade retrieved documents
  - Trigger web search if poor
- **ReAct for RAG**
  - Reasoning + retrieval actions
  - Multi-hop question answering

## 6.7 RAG Evaluation

### 6.7.1 Retrieval Metrics
- **Recall@k**
  - Proportion of relevant docs in top-k
  - Critical for RAG success
- **MRR (Mean Reciprocal Rank)**
  - 1/rank of first relevant document
- **NDCG (Normalized Discounted Cumulative Gain)**
  - Accounts for graded relevance
  - Position-aware
- **Hit Rate**
  - At least one relevant in top-k

### 6.7.2 Generation Metrics
- **Faithfulness**
  - Is generated content supported by context?
  - NLI-based evaluation
  - Claim-evidence matching
- **Answer Relevance**
  - Does answer address the question?
  - Semantic similarity metrics
- **Context Relevance**
  - Are retrieved documents relevant to query?
  - Query-document similarity

### 6.7.3 End-to-End Evaluation
- **RAGAS Framework**
  - Faithfulness, answer relevancy, context precision, context recall
- **LLM-as-Judge**
  - GPT-4 for evaluation
  - Consistency concerns
- **Human Evaluation**
  - Ground truth answers
  - Annotated relevance judgments
- **A/B Testing**
  - Online evaluation
  - User satisfaction metrics

### 6.7.4 Benchmarks
- **Natural Questions**
  - Real Google queries
  - Wikipedia answers
- **HotpotQA**
  - Multi-hop reasoning
  - Distractor documents
- **MS MARCO**
  - Bing search queries
  - Large-scale retrieval

### 🔴 Commonly Missed / Underrated Topics (RAG)

- **Retrieval Failure Modes**
  - Query-document vocabulary mismatch
  - Semantic drift in long documents
  - Edge case queries with no relevant docs
- **Chunk Boundary Problems**
  - Information split across chunks
  - Context loss at boundaries
  - Overlap strategies and limitations
- **Embedding Model Selection**
  - Domain mismatch (general vs domain-specific)
  - Multilingual considerations
  - Instruction-tuned embeddings
- **Vector Database Tradeoffs**
  - Recall vs latency tradeoffs
  - Memory vs disk storage
  - Consistency requirements
- **Citation and Attribution**
  - Teaching LLM to cite sources
  - Verifying citations are correct
  - Handling multiple supporting sources
- **Dynamic Knowledge Updates**
  - Incremental indexing
  - Document versioning
  - Deletion and modification handling
- **RAG Security**
  - Prompt injection via documents
  - Data leakage risks
  - Access control in multi-tenant RAG
- **Cost Optimization**
  - Embedding API costs at scale
  - Caching strategies
  - Tiered retrieval (cheap → expensive)
- **Multimodal RAG**
  - Image + text retrieval
  - CLIP embeddings
  - Document understanding with layout
- **Evaluation Gaps**
  - Training/test set leakage in benchmarks
  - Synthetic evaluation limitations
  - Real-world query distribution mismatch

### 🎯 Interview Focus (RAG)

**Beginner Level:**
- Explain RAG pipeline end-to-end
- Difference between dense and sparse retrieval
- Why chunk documents?
- Common vector database options

**Intermediate Level:**
- Compare bi-encoder vs cross-encoder
- Chunking strategies and tradeoffs
- ANN algorithms (HNSW, IVF)
- Reranking pipeline design
- Handle "lost in the middle" problem

**Senior Level:**
- Design RAG system for 10M documents
- Optimize latency for real-time RAG
- Implement self-correcting RAG
- Design evaluation framework
- Handle multimodal retrieval

**Differentiating Questions:**
- "Your RAG system retrieves irrelevant documents for domain-specific queries — what's wrong?" (Answer: embedding domain mismatch, need domain-adapted embeddings)
- "Design RAG that can answer across 100M documents in <100ms" (Answer: hierarchical retrieval, aggressive filtering, caching, approximate methods)
- "How do you detect when RAG should abstain?" (Answer: retrieval confidence threshold, answer uncertainty, no relevant docs detected)

---

# 7. Deployment & MLOps

## 7.1 Model Deployment

### 7.1.1 Deployment Patterns
- **Batch Prediction**
  - Process data in bulk
  - Schedule-based or trigger-based
  - High throughput, no latency constraints
- **Real-time/Synchronous**
  - API endpoint
  - Low latency requirements
  - Request-response pattern
- **Streaming**
  - Event-driven processing
  - Kafka, Kinesis pipelines
  - Near real-time
- **Edge Deployment**
  - On-device inference
  - Resource constraints
  - Offline capability

### 7.1.2 Model Serving
- **REST API**
  - HTTP-based, universal
  - JSON payload
  - Frameworks: Flask, FastAPI
- **gRPC**
  - Binary protocol, faster
  - Strongly typed
  - Better for internal services
- **Model Servers**
  - TensorFlow Serving
  - TorchServe
  - NVIDIA Triton
  - BentoML
- **Serverless**
  - AWS Lambda, Google Cloud Functions
  - Auto-scaling, pay-per-use
  - Cold start considerations

### 7.1.3 Scaling Strategies
- **Horizontal Scaling**
  - Multiple model instances
  - Load balancing
  - Kubernetes deployments
- **Vertical Scaling**
  - Larger instances
  - GPU acceleration
- **Auto-scaling**
  - CPU/GPU utilization triggers
  - Request queue depth
  - Custom metrics

## 7.2 Model Optimization

### 7.2.1 Quantization
- **Post-Training Quantization**
  - INT8, INT4 weights
  - Calibration dataset
  - Accuracy recovery techniques
- **Quantization-Aware Training (QAT)**
  - Simulate quantization during training
  - Better accuracy preservation
- **GPTQ, AWQ, GGUF**
  - LLM-specific quantization
  - Layer-wise quantization
  - Mixed precision

### 7.2.2 Pruning
- **Magnitude Pruning**
  - Remove small weights
  - Structured vs unstructured
- **Movement Pruning**
  - Learn which weights to keep
- **Lottery Ticket Hypothesis**
  - Find sparse subnetworks
- **Impact on Inference**
  - Unstructured: memory savings
  - Structured: speedup

### 7.2.3 Knowledge Distillation
- **Teacher-Student Training**
  - Large teacher, small student
  - Soft targets with temperature
- **Distillation Loss**
  - KL divergence from teacher
  - Combined with ground truth
- **Applications**
  - BERT → DistilBERT
  - GPT → Smaller student

### 7.2.4 Compilation and Optimization
- **ONNX**
  - Framework-agnostic format
  - Runtime optimization
- **TensorRT**
  - NVIDIA GPU optimization
  - Layer fusion, precision calibration
- **OpenVINO**
  - Intel hardware optimization
- **TVM**
  - Compiler stack for diverse hardware

## 7.3 MLOps Pipeline

### 7.3.1 Experiment Tracking
- **Tools**
  - Weights & Biases
  - MLflow
  - TensorBoard
  - Neptune
- **Tracking Metrics**
  - Loss curves, evaluation metrics
  - Hyperparameters
  - Artifacts (models, datasets)
- **Experiment Reproducibility**
  - Seed setting
  - Dependency pinning
  - Environment versioning

### 7.3.2 Versioning
- **Code Versioning**
  - Git workflows
  - Branch strategies
- **Data Versioning**
  - DVC (Data Version Control)
  - LakeFS
  - Pachyderm
- **Model Versioning**
  - Model registry
  - Stage transitions (staging → production)
  - Artifact metadata

### 7.3.3 CI/CD for ML
- **Continuous Integration**
  - Automated testing
  - Code quality checks
  - Unit tests for data pipelines
- **Continuous Training (CT)**
  - Retraining triggers
  - Data drift detection
  - Performance degradation
- **Continuous Deployment**
  - Automated model promotion
  - Canary deployments
  - Blue-green deployments
- **Testing**
  - Model performance tests
  - Integration tests
  - Data validation tests

### 7.3.4 Pipeline Orchestration
- **Workflow Tools**
  - Apache Airflow
  - Kubeflow Pipelines
  - Prefect
  - Dagster
- **Pipeline Components**
  - Data extraction
  - Feature engineering
  - Training
  - Evaluation
  - Deployment
- **Scheduling**
  - Cron-based
  - Event-driven
  - Dependency management

## 7.4 Monitoring

### 7.4.1 Model Performance Monitoring
- **Accuracy Metrics**
  - Track prediction accuracy over time
  - Ground truth availability delay
- **Prediction Distribution**
  - Output distribution drift
  - Confidence score trends
- **Business Metrics**
  - Conversion rates
  - Revenue impact
  - User engagement

### 7.4.2 Data Drift Detection
- **Feature Drift**
  - Statistical tests: KS test, PSI
  - Distribution comparison
- **Concept Drift**
  - Relationship change: X → Y
  - Performance degradation
- **Detection Methods**
  - Statistical: KS, Chi-square, PSI
  - Model-based: classifier drift detection
  - Time window comparisons
- **Response Strategies**
  - Alerting
  - Automatic retraining
  - Model fallback

### 7.4.3 Infrastructure Monitoring
- **Latency**
  - P50, P95, P99 response times
  - SLA compliance
- **Throughput**
  - Requests per second
  - Resource utilization
- **Error Rates**
  - 4xx, 5xx errors
  - Model inference errors
- **Resource Metrics**
  - CPU, GPU, memory usage
  - Disk I/O
  - Network throughput

### 7.4.4 Logging and Observability
- **Structured Logging**
  - JSON format
  - Correlation IDs
- **Distributed Tracing**
  - Request flow visualization
  - Jaeger, Zipkin
- **Dashboards**
  - Grafana, CloudWatch
  - Custom ML dashboards

## 7.5 GenAI-Specific Deployment

### 7.5.1 LLM Deployment Challenges
- **High Memory Requirements**
  - GPU memory constraints
  - Model sharding strategies
- **Variable Output Length**
  - Unpredictable latency
  - Streaming responses
- **Cost Management**
  - Token-based pricing
  - Caching strategies
  - Request batching

### 7.5.2 LLM Serving Optimization
- **vLLM**
  - PagedAttention
  - Continuous batching
  - High throughput
- **Text Generation Inference (TGI)**
  - HuggingFace optimized server
  - Flash Attention support
- **OpenAI-Compatible APIs**
  - Standard interface
  - Easy migration

### 7.5.3 Guardrails
- **Input Validation**
  - Prompt injection detection
  - Content policy enforcement
- **Output Filtering**
  - Toxicity detection
  - PII detection and masking
- **Rate Limiting**
  - Prevent abuse
  - Fair usage

### 7.5.4 RAG Production Considerations
- **Embedding Pipeline**
  - Batch embedding jobs
  - Incremental updates
- **Vector Store Operations**
  - Backup and recovery
  - Index optimization
- **Query Caching**
  - Common query results
  - Embedding cache

## 7.6 Security and Compliance

### 7.6.1 Model Security
- **Model Inversion Attacks**
  - Reconstruct training data
  - Membership inference
- **Adversarial Examples**
  - Evasion attacks
  - Defenses: adversarial training
- **Model Extraction**
  - Stealing model functionality
  - Rate limiting, watermarking
- **Supply Chain Security**
  - Dependency vulnerabilities
  - Model provenance

### 7.6.2 Data Privacy
- **PII Handling**
  - Detection and redaction
  - Differential privacy
- **Federated Learning**
  - Train without centralizing data
  - Secure aggregation
- **Regulatory Compliance**
  - GDPR: right to explanation
  - CCPA: data deletion
  - HIPAA for healthcare

### 7.6.3 Responsible AI
- **Fairness**
  - Bias detection
  - Fairness metrics
  - Mitigation techniques
- **Explainability**
  - SHAP, LIME
  - Attention visualization
  - Counterfactual explanations
- **Transparency**
  - Model cards
  - Data sheets
  - Documentation

### 🔴 Commonly Missed / Underrated Topics (Deployment & MLOps)

- **Shadow Deployment**
  - Test new model with production traffic
  - No user impact
  - Compare predictions
- **Champion/Challenger Pattern**
  - Current champion vs new challenger
  - Statistical significance testing
  - Gradual rollout
- **Cold Start Problems**
  - New user/item recommendations
  - Fallback strategies
- **Training-Serving Skew**
  - Different code paths
  - Different data processing
  - Prevention strategies
- **Feature Store**
  - Centralized feature management
  - Online vs offline consistency
  - Feast, Tecton
- **Model Cards**
  - Document model characteristics
  - Intended use, limitations
  - Ethical considerations
- **A/B Testing for ML**
  - Randomization units
  - Sufficient sample size
  - Multiple comparison correction
- **Online Learning**
  - Continuous model updates
  - Stability challenges
  - FTRL, adaptive learning rates
- **Multi-Model Endpoints**
  - Serve multiple models from one endpoint
  - Cost efficiency
- **Disaster Recovery**
  - Model backup strategies
  - Rollback procedures
  - RTO/RPO planning

### 🎯 Interview Focus (Deployment & MLOps)

**Beginner Level:**
- REST vs gRPC for model serving
- What is model drift and how to detect?
- CI/CD basics for ML
- Monitoring metrics for deployed models

**Intermediate Level:**
- Design model deployment pipeline
- Handle training-serving skew
- Implement A/B testing for models
- Optimize model for edge deployment
- Design data versioning strategy

**Senior Level:**
- Design MLOps platform for enterprise
- Architect multi-region model serving
- Implement automated retraining pipeline
- Design disaster recovery for ML systems
- Optimize LLM serving at scale

**Differentiating Questions:**
- "Your model accuracy drops 5% in production — debugging approach?" (Answer: data drift check, training-serving skew, feature pipeline bugs)
- "Design system to serve 10K RPS with 100ms latency" (Answer: caching, batching, model optimization, horizontal scaling)
- "How to handle ground truth delay in monitoring?" (Answer: proxy metrics, prediction distribution monitoring, delayed feedback loops)

---

# 🔗 Cross-Domain Connections

## Statistics ↔ Machine Learning
- **Loss Functions as Statistical Estimators**
  - MSE ↔ Gaussian MLE
  - Cross-entropy ↔ Bernoulli/Multinomial MLE
  - MAE ↔ Laplace MLE
- **Regularization as Priors**
  - L2 ↔ Gaussian prior (Ridge = MAP with Gaussian)
  - L1 ↔ Laplace prior (Lasso = MAP with Laplace)
- **Bootstrap and Bagging**
  - Bootstrap sampling foundation of bagging
  - Random Forest = Bootstrap + Random Subspace
- **Bayesian Methods in ML**
  - Bayesian optimization for hyperparameters
  - Bayesian neural networks
  - Gaussian Processes

## Deep Learning ↔ NLP/LLMs
- **Embeddings Across Domains**
  - Word2Vec → Sentence-BERT → LLM token embeddings
  - Continuous progression in representation learning
- **Attention as Universal Mechanism**
  - Started in NLP (machine translation)
  - Applied to Vision (ViT)
  - Now core to all modalities
- **Transfer Learning Evolution**
  - ImageNet pre-training → BERT pre-training → LLM pre-training
  - Same paradigm, different modalities
- **CNN → RNN → Transformer Progression**
  - Architectural evolution in NLP
  - Now Transformers dominate both NLP and Vision

## LLMs ↔ RAG
- **Complementary Strengths**
  - LLM: reasoning, fluency, general knowledge
  - RAG: factual accuracy, up-to-date info, attribution
- **Embedding Connection**
  - LLM embeddings power RAG retrieval
  - Fine-tuned LLMs improve embeddings
- **Prompt Engineering Bridge**
  - RAG prompt design crucial for LLM utilization
  - LLM capabilities determine RAG effectiveness
- **Evaluation Interdependence**
  - RAG success depends on LLM generation quality
  - LLM evaluation includes RAG scenarios

## MLOps ↔ All Domains
- **Experiment Tracking**
  - Essential for all model development
  - Reproducibility across ML, DL, NLP, LLMs
- **Monitoring**
  - Different metrics per domain
  - Same infrastructure principles
- **Deployment Patterns**
  - Batch vs real-time applies universally
  - Scaling strategies domain-agnostic
- **Versioning**
  - Code, data, model for all domains
  - Especially critical for LLMs (prompt versioning)

---

# 🧠 Learning Gaps Checklist

## Fundamentals (Often Rushed)
- [ ] Derive MLE for common distributions from scratch
- [ ] Implement backpropagation manually for simple network
- [ ] Prove properties of matrix decompositions
- [ ] Understand all assumptions of linear regression
- [ ] Derive gradient descent convergence rates

## Practical Skills (Often Neglected)
- [ ] Debug data leakage in a real pipeline
- [ ] Handle 99:1 class imbalance effectively
- [ ] Optimize inference latency by 10×
- [ ] Set up proper cross-validation for time series
- [ ] Implement custom loss function with autograd

## Advanced Topics (Often Skipped)
- [ ] Understand NTK and infinite-width networks
- [ ] Implement LoRA from scratch
- [ ] Design distributed training for large models
- [ ] Build RAG system with evaluation framework
- [ ] Set up end-to-end MLOps pipeline

## Interview-Specific (Commonly Stumbled)
- [ ] Explain p-value without errors
- [ ] Compare optimizers with mathematical intuition
- [ ] Debug model performance degradation
- [ ] Design A/B test with proper statistics
- [ ] Handle adversarial questions about your approach

## Production Readiness (Critical Gap)
- [ ] Design monitoring for model drift
- [ ] Implement rollback strategies
- [ ] Handle training-serving skew detection
- [ ] Design for failure modes and edge cases
- [ ] Balance latency, cost, and accuracy

## Emerging Areas (Stay Current)
- [ ] Understand MoE architecture tradeoffs
- [ ] Implement speculative decoding
- [ ] Design multi-modal RAG systems
- [ ] Understand alignment beyond RLHF
- [ ] Navigate LLM safety and red teaming

---

> **Total Concepts Covered:** 400+ nodes across 7 major domains
> 
> **Preparation Strategy:** Start with fundamentals, progress through intermediate, master senior-level system design and tradeoff analysis
> 
> **Practice Recommendation:** For each concept, be able to explain to a non-technical stakeholder, implement a simplified version, and discuss real-world tradeoffs

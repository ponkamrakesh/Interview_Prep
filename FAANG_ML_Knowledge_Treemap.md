# 🎓 FAANG-Level ML/AI Interview Knowledge Treemap
*Comprehensive hierarchical guide covering 1000+ essential concepts for Data Scientists, ML Engineers, and GenAI Engineers*

---

# 1. Machine Learning

## 1.1 Fundamentals & Core Concepts

### 1.1.1 Problem Formulation
#### 1.1.1.1 Supervised Learning [F]
##### Task formulation
##### Input-output relationship
##### Objective function design
##### Loss function selection

#### 1.1.1.2 Unsupervised Learning [F]
##### Clustering objectives
##### Dimensionality reduction goals
##### Density estimation
##### Anomaly detection framing

#### 1.1.1.3 Semi-Supervised Learning [A]
##### Pseudo-labeling
##### Consistency regularization
##### Entropy minimization
##### Self-training variants

#### 1.1.1.4 Reinforcement Learning Basics [A]
##### Markov Decision Processes
##### Policy vs value-based methods
##### Exploration-exploitation tradeoff
##### Reward shaping

### 1.1.2 Bias-Variance Decomposition [F] [T]
#### 1.1.2.1 Core Concepts
##### Bias definition (irreducible error)
##### Variance definition (model sensitivity)
##### Tradeoff visualization
##### Bayes optimal error

#### 1.1.2.2 Bias Sources
##### Model class limitations
##### Optimization limitations
##### Regularization effects
##### Training procedure bias

#### 1.1.2.3 Variance Sources
##### Dataset sensitivity
##### Feature correlations
##### Model capacity
##### Noise amplification

#### 1.1.2.4 Decomposition Extensions
##### Bias-variance-covariance
##### Bias-variance in ensemble methods
##### Bias-variance in neural networks
##### Connection to generalization bounds

#### 1.1.2.5 Practical Implications [F]
##### High bias indicators
##### High variance indicators
##### Underfitting vs overfitting
##### Learning curves interpretation

### 1.1.3 Fundamental Trade-offs
#### 1.1.3.1 Accuracy vs Interpretability [F] [A]
##### Model complexity spectrum
##### Black-box vs explainable models
##### Approximation quality
##### Local vs global explanations

#### 1.1.3.2 Speed vs Accuracy [F] [S]
##### Inference latency constraints
##### Model compression techniques
##### Batch vs real-time processing
##### Hardware-software trade-offs

#### 1.1.3.3 Memory vs Computation [A] [S]
##### Space-time complexity
##### Streaming algorithms
##### Distributed vs centralized
##### Hardware utilization

#### 1.1.3.4 Data Quality vs Quantity [F]
##### Label noise tolerance
##### Sample efficiency
##### Active learning
##### Data valuation

---

## 1.2 Supervised Learning Algorithms

### 1.2.1 Linear Models [F]
#### 1.2.1.1 Linear Regression
##### Ordinary least squares (OLS)
##### Normal equation
##### Gradient descent variants
##### Closed-form solutions
##### Assumptions (linearity, independence, homoscedasticity)
##### Durbin-Watson test
##### Residual analysis

#### 1.2.1.2 Regularized Linear Regression [F]
##### Ridge regression (L2)
##### Lasso regression (L1)
##### ElasticNet (L1+L2)
##### Elastic net path
##### Coordinate descent
##### Orthogonal matching pursuit
##### Shooting algorithm

#### 1.2.1.3 Logistic Regression [F]
##### Binary classification
##### Multi-class extensions (softmax)
##### Sigmoid function properties
##### Log-odds interpretation
##### Decision boundary
##### Probability calibration
##### Separation margins

#### 1.2.1.4 Generalized Linear Models [T]
##### Exponential family
##### Link functions
##### Canonical form
##### Variance function
##### Poisson regression
##### Negative binomial regression

### 1.2.2 Distance-Based Methods [F]
#### 1.2.2.1 k-Nearest Neighbors (KNN) [F]
##### Distance metrics (Euclidean, Manhattan, Cosine, Hamming)
##### k selection (bias-variance)
##### Computational complexity
##### Curse of dimensionality
##### Weighted KNN variants
##### Approximate nearest neighbors (KD-trees, Ball trees)
##### Locality-sensitive hashing (LSH)

#### 1.2.2.2 Support Vector Machines (SVM) [F] [A]
##### Margin maximization
##### Soft margin formulation
##### Kernel trick fundamentals
##### Support vectors
##### Dual problem formulation
##### Kernel types (linear, polynomial, RBF, sigmoid)
##### One-vs-rest multiclass
##### One-vs-one multiclass
##### Probability calibration (Platt scaling)

### 1.2.3 Tree-Based Methods [F]
#### 1.2.3.1 Decision Trees [F]
##### Splitting criteria (Gini, Entropy, Information Gain)
##### Greedy construction
##### Tree depth control
##### Feature importance (mean decrease in impurity)
##### Handling categorical variables
##### Missing value strategies
##### Surrogate splits

#### 1.2.3.2 Ensemble Methods [F] [A]
##### Bagging principles
##### Out-of-bag error estimation
##### Bootstrap samples
##### Variance reduction mechanism

##### Random Forests [F]
###### Feature subsampling
###### Tree diversity
###### Parallel training
###### Variable importance
###### Permutation importance
###### Proximity matrix

##### Boosting [F] [A]
###### Adaptive boosting (AdaBoost)
###### Gradient boosting machines (GBM)
###### Forward stagewise additive modeling
###### Loss function flexibility

##### XGBoost [F] [A]
###### Second-order approximation
###### Column subsampling
###### Row subsampling
###### Regularization (L1, L2)
###### Sparsity-aware split finding
###### Weighted quantile sketch
###### Tree pruning

##### LightGBM [A]
###### Leaf-wise growth
###### Histogram-based learning
###### GOSS (Gradient-based One-Side Sampling)
###### EFB (Exclusive Feature Bundling)
###### Parallel learning
###### GPU acceleration

##### CatBoost [A]
###### Categorical feature handling
###### Ordered boosting
###### Symmetric trees
###### Feature interaction importance

#### 1.2.3.3 Stacking & Blending [A]
##### Meta-learner architecture
##### Cross-validation folds
##### Feature space creation
##### Stacking generalization
##### Level-0 vs level-1 models
##### Soft vs hard voting

---

## 1.3 Unsupervised Learning

### 1.3.1 Clustering [F]
#### 1.3.1.1 Partitioning Methods [F]
##### k-Means [F]
###### Lloyd's algorithm
###### Initialization strategies (k-means++, random, hierarchical)
###### Convergence criteria
###### Elbow method
###### Silhouette analysis
###### Gap statistic
###### Mini-batch k-means

##### k-Medoids
###### Partitioning around medoids (PAM)
###### Robustness to outliers
###### Computational cost

##### k-Modes
###### Categorical clustering
###### Mode update
###### Dissimilarity metrics

#### 1.3.1.2 Hierarchical Clustering [F]
##### Agglomerative (bottom-up)
##### Divisive (top-down)
##### Linkage criteria
###### Single linkage (nearest neighbor)
###### Complete linkage (furthest neighbor)
###### Average linkage (UPGMA)
###### Ward linkage (minimum variance)
###### Centroid linkage

##### Dendrogram interpretation
##### Cutting strategies
##### Cophenetic correlation

#### 1.3.1.3 Density-Based Clustering [A]
##### DBSCAN [F]
###### Epsilon-neighborhood
###### Core points, border points
###### Density reachability
###### eps and minPts selection
###### Handling varying densities

##### OPTICS
###### Reachability distance
###### Ordering
###### Density-based hierarchy

##### Isolation Forest [A] [S]
###### Anomaly isolation
###### Anomaly score calculation
###### Contamination parameter

#### 1.3.1.4 Probabilistic Clustering [T] [A]
##### Gaussian Mixture Models (GMM)
###### Mixture components
###### EM algorithm
###### Model selection (BIC, AIC)
###### Covariance structures (spherical, diagonal, full)
###### Singularity handling

##### Bayesian Mixture Models
###### Dirichlet process mixtures
###### Infinite mixture models
###### Chinese restaurant process

#### 1.3.1.5 Clustering Evaluation [F]
##### Silhouette coefficient
##### Davies-Bouldin index
##### Calinski-Harabasz index
##### Intra-cluster vs inter-cluster distances
##### Purity (when labels available)
##### Adjusted Rand Index
##### Normalized Mutual Information

### 1.3.2 Dimensionality Reduction [F]
#### 1.3.2.1 Linear Methods [F]
##### Principal Component Analysis (PCA) [F]
###### Eigenvalue decomposition
###### Variance explanation
###### Scree plot
###### Cumulative variance
###### Singular value decomposition (SVD)
###### Centering and scaling
###### Biplot interpretation
###### Procrustes analysis

##### Factor Analysis [T]
###### Latent variables
###### Factor loadings
###### Communalities
###### Specific variance
###### Rotation methods (orthogonal, oblique)

##### Independent Component Analysis (ICA) [T]
###### Statistical independence
###### Non-Gaussian assumption
###### Blind source separation
###### Kurtosis-based contrast

##### Canonical Correlation Analysis (CCA) [T]
###### Multi-view learning
###### Canonical variates
###### Correlation maximization

#### 1.3.2.2 Non-Linear Methods [A]
##### Manifold Learning [A]
###### Intrinsic dimensionality
###### Isometric mapping (Isomap)
###### Locally linear embedding (LLE)
###### Laplacian eigenmaps
###### Diffusion maps
###### UMAP
###### t-SNE [F]

##### Kernel PCA [A]
###### Kernel trick application
###### Non-linear variance
###### Kernel selection
###### Pre-image problem

##### Autoencoders [A]
###### Encoder-decoder architecture
###### Bottleneck representation
###### Reconstruction loss
###### Variational autoencoders (VAE)

#### 1.3.2.3 Dimensionality Reduction Trade-offs
##### Information loss
##### Computational cost
##### Interpretability vs compression
##### Visualization vs utility

---

## 1.4 Feature Engineering & Selection

### 1.4.1 Feature Engineering [F] [S]
#### 1.4.1.1 Numerical Features [F]
##### Scaling methods
###### Standardization (z-score)
###### Min-max scaling
###### Robust scaling (quantile-based)
###### Log transformation
###### Box-Cox transformation
###### Yeo-Johnson transformation
###### Power transformations

##### Binning & Discretization
###### Equal-width binning
###### Equal-frequency binning
###### Quantile-based binning
###### Decision tree-based binning
###### Domain-driven binning

##### Polynomial features
###### Interaction terms
###### Squared terms
###### Basis expansion
###### Spline features

##### Domain-specific transformations
###### Temporal features (seasonality, trends)
###### Geographic features (distance, clustering)
###### Financial ratios

#### 1.4.1.2 Categorical Features [F]
##### Encoding schemes
###### One-hot encoding
###### Ordinal encoding
###### Binary encoding
###### Frequency encoding
###### Helmert encoding
###### Target encoding [A]
###### Likelihood encoding
###### Weight of Evidence (WOE)

##### High cardinality handling
###### Grouping rare categories
###### Hashing trick
###### Embedding-based approaches
###### Target-based grouping

##### Ordinal feature handling
###### Preserving order
###### Distance-based encoding

#### 1.4.1.3 Text Features [F]
##### Bag of words (BoW)
##### TF-IDF
##### N-grams
##### Character n-grams
##### Word embeddings (Word2Vec, GloVe, FastText)
##### Subword tokenization
##### Contextual embeddings

#### 1.4.1.4 Date/Time Features [F] [S]
##### Temporal decomposition
###### Year, month, day extraction
###### Day of week
###### Hour, minute, second
###### Quarter, semester
###### Cyclical encoding (sine/cosine)

##### Relative time features
###### Time since first event
###### Days until event
###### Time between events
###### Seasonal indicators

#### 1.4.1.5 Feature Cross & Interaction [A]
##### Statistical interaction detection
##### Domain-driven interactions
##### Multiplicative combinations
##### High-order interactions

#### 1.4.1.6 Automated Feature Engineering [A]
##### Featuretools
##### TSFRESH
##### AutoML feature discovery
##### Neural architecture search for features

### 1.4.2 Feature Selection [F] [A]
#### 1.4.2.1 Filter Methods [F]
##### Correlation-based
###### Pearson correlation
###### Spearman correlation
###### Kendall tau
###### Mutual information
###### Distance correlation

##### Statistical tests
###### Chi-square test (categorical)
###### ANOVA F-test (numerical)
###### Kruskal-Wallis test
###### Variance threshold

##### Information-theoretic measures
###### Information gain
###### Gain ratio
###### Gini index

#### 1.4.2.2 Wrapper Methods [F]
##### Forward selection
##### Backward elimination
##### Bidirectional elimination
##### Recursive feature elimination (RFE)
##### Sequential floating selection
##### Computational cost considerations

#### 1.4.2.3 Embedded Methods [F] [A]
##### L1 regularization (Lasso)
##### Tree-based importance
###### Mean decrease in impurity
###### Permutation importance
###### SHAP feature importance

##### Stability selection [A]
##### Elastic net paths

#### 1.4.2.4 Advanced Selection [A]
##### Boruta algorithm
##### Hybrid methods
##### Multi-objective optimization
##### Domain expert guidance

#### 1.4.2.5 Feature Selection Pitfalls [F] [A]
##### Data leakage through feature selection
##### Multiple hypothesis testing corrections
##### Overfitting to validation set
##### Instability under data perturbation
##### Target-dependent bias

---

## 1.5 Regularization Techniques [F] [A] [T]

### 1.5.1 L1 & L2 Regularization [F]
#### 1.5.1.1 L1 Regularization (Lasso) [F]
##### Sparsity-inducing property
##### Geometry of L1 ball
##### Feature selection capability
##### Non-differentiability at zero
##### Soft thresholding
##### Proximal algorithms

#### 1.5.1.2 L2 Regularization (Ridge) [F]
##### Weight decay interpretation
##### Gaussian prior interpretation
##### Quadratic penalty
##### Analytical solution existence
##### Eigenvalue dampening

#### 1.5.1.3 ElasticNet [A]
##### Hybrid L1+L2
##### Alpha and lambda parameters
##### Grouping effect
##### Naïve vs true elastic net

### 1.5.2 Early Stopping [F]
#### 1.5.2.1 Validation-based stopping
##### Patience parameter
##### Window-based stopping
##### Smooth stopping

#### 1.5.2.2 Stopping criteria
##### Loss-based
##### Metric-based (accuracy, F1, etc.)
##### Gradient-based

### 1.5.3 Dropout & Stochastic Regularization [F] [A]
#### 1.5.3.1 Dropout mechanisms
##### Bernoulli dropout
##### Variational dropout
##### Spatial dropout
##### DropConnect
##### Concrete dropout

#### 1.5.3.2 Dropout interpretation
##### Approximate Bayesian inference
##### Ensemble averaging
##### Co-adaptation prevention

### 1.5.4 Data Augmentation [F]
#### 1.5.4.1 Image augmentation
##### Geometric transformations (rotation, flip, crop)
##### Color jittering
##### Cutout & CutMix
##### MixUp
##### AutoAugment
##### RandAugment

#### 1.5.4.2 Text augmentation
##### Back-translation
##### Paraphrasing
##### Token shuffling
##### EDA (Easy Data Augmentation)

#### 1.5.4.3 Time series augmentation
##### Jittering
##### Scaling
##### Rotation
##### Magnitude warping
##### Time warping

### 1.5.5 Batch Normalization [F] [A]
#### 1.5.5.1 Internal covariate shift
##### Definition and impact
##### Batch statistics
##### Layer normalization
##### Instance normalization
##### Group normalization

#### 1.5.5.2 Batch norm mechanics
##### Forward pass (training)
##### Forward pass (inference)
##### Exponential moving average
##### Gamma and beta parameters
##### Momentum parameter

#### 1.5.5.3 Batch norm variants
##### Layer normalization
##### Batch renormalization
##### EvoNorm
##### FilterResponse normalization

### 1.5.6 Advanced Regularization [A]
#### 1.5.6.1 Weight noise injection
##### Gaussian weight noise
##### Dropout weight variance
##### Information bottleneck

#### 1.5.6.2 Mixup & CutMix [A]
##### Linear interpolation
##### Manifold mixing
##### Region-based mixing

#### 1.5.6.3 Adversarial training [A]
##### Adversarial examples
##### FGSM (Fast Gradient Sign Method)
##### PGD (Projected Gradient Descent)
##### Robust optimization

---

## 1.6 Optimization & Learning

### 1.6.1 Optimization Fundamentals [F] [T]
#### 1.6.1.1 Convex Optimization [F] [T]
##### Convex sets and functions
##### Convex vs non-convex problems
##### Global vs local minima
##### Optimality conditions (KKT, Lagrange)
##### Feasibility vs optimality

#### 1.6.1.2 Gradient-Based Optimization [F]
##### Gradient descent
##### Learning rate selection
##### Convergence guarantees
##### Lipschitz smoothness
##### Strong convexity

#### 1.6.1.3 Stochastic Optimization [F] [T]
##### Stochastic gradient descent (SGD)
##### Mini-batch sampling
##### Noise characteristics
##### Variance reduction
##### Convergence analysis

### 1.6.2 Gradient Descent Variants [F]
#### 1.6.2.1 Momentum-Based Methods [F]
##### Momentum (Heavy Ball)
##### Polyak averaging
##### Nesterov accelerated gradient
##### Acceleration mechanisms
##### Momentum scheduling

#### 1.6.2.2 Adaptive Methods [F]
##### AdaGrad [F]
###### Per-feature learning rates
###### Accumulated gradient squares
###### Sparse update advantage

##### RMSProp [F]
###### Exponential moving average
###### Adaptive learning rates
###### Centered variant

##### Adam [F] [A]
###### Adaptive moments (first & second)
###### Bias correction
###### Effective learning rate
###### Convergence issues [A]
###### Empirical improvements (AdamW, AMSGrad)

##### Variants [A]
###### AdamW (decoupled weight decay)
###### RAdam (warm-up adaptive momentum)
###### AdaBound
###### LookaHead optimizer

#### 1.6.2.3 Second-Order Methods [T] [A]
##### Newton's method
##### Quasi-Newton methods (BFGS, L-BFGS)
##### Hessian approximation
##### Computational complexity
##### Limited-memory variants

### 1.6.3 Learning Rate Scheduling [F]
#### 1.6.3.1 Scheduling strategies
##### Step decay
##### Exponential decay
##### Polynomial decay
##### Cosine annealing
##### Warm restarts (SGDR)
##### Cyclic learning rates
##### Learning rate warmup [A]

#### 1.6.3.2 Learning rate search [F]
##### Grid search
##### Random search
##### Bayesian optimization
##### Cyclic learning rate (finding optimal range)

### 1.6.4 Optimization Challenges [A]
#### 1.6.4.1 Saddle points
##### Definition and prevalence
##### Escape dynamics
##### Second-order escape

#### 1.6.4.2 Plateaus and local minima
##### Mode connectivity
##### Loss landscape visualization
##### Lottery ticket hypothesis

#### 1.6.4.3 Overfitting dynamics [A]
##### Early stopping windows
##### Generalization gap
##### Sharp vs flat minima
##### SAM (Sharpness Aware Minimization)

---

## 1.7 Validation & Evaluation

### 1.7.1 Data Splitting Strategies [F]
#### 1.7.1.1 Train-Validation-Test Split [F]
##### Temporal ordering (time series)
##### Stratification (imbalanced data)
##### Random sampling
##### Repeatable splits (random seed)

#### 1.7.1.2 Cross-Validation [F]
##### k-Fold cross-validation
##### Stratified k-fold
##### Time series cross-validation
##### Leave-one-out (LOO)
##### Nested cross-validation [A]
##### Group k-fold (for grouped data)
##### Adversarial validation [A]

### 1.7.2 Classification Metrics [F]
#### 1.7.2.1 Binary Classification [F]
##### Confusion matrix
##### Accuracy [F]
##### Precision [F]
##### Recall / Sensitivity [F]
##### Specificity
##### F1 Score [F]
##### ROC-AUC [F] [A]
##### PR-AUC [A]
##### Matthews Correlation Coefficient
##### Cohen's Kappa

#### 1.7.2.2 Multi-Class Classification [F]
##### Macro-averaging
##### Micro-averaging
##### Weighted averaging
##### Per-class metrics
##### One-vs-rest approach

#### 1.7.2.3 Imbalanced Classification [F] [A]
##### Class weights
##### Sampling strategies
##### Threshold adjustment
##### Anomaly detection framing
##### Cost-sensitive learning

### 1.7.3 Regression Metrics [F]
#### 1.7.3.1 Point Prediction Metrics [F]
##### Mean Absolute Error (MAE)
##### Mean Squared Error (MSE) / RMSE
##### Mean Absolute Percentage Error (MAPE)
##### Median Absolute Error
##### R² Score
##### Adjusted R²

#### 1.7.3.2 Distribution-Based Metrics [A]
##### Quantile loss
##### Pinball loss
##### Interval score (probabilistic)
##### Continuous Ranked Probability Score (CRPS)

#### 1.7.3.3 Domain-Specific Metrics [A]
##### RMSLE (log-scale errors)
##### Symmetric MAPE
##### Huber loss
##### Tukey's biweight

### 1.7.4 Ranking & Recommendation Metrics [A]
#### 1.7.4.1 Ranking metrics
##### NDCG (Normalized Discounted Cumulative Gain)
##### MRR (Mean Reciprocal Rank)
##### MAP (Mean Average Precision)
##### Hit Rate

#### 1.7.4.2 Recommendation metrics
##### Precision@K
##### Recall@K
##### Diversity metrics
##### Novelty metrics

### 1.7.5 Statistical Significance Testing [F] [A]
#### 1.7.5.1 Hypothesis testing
##### Null and alternative hypotheses
##### Type I and Type II errors
##### Power and sample size
##### Multiple comparison corrections (Bonferroni, FDR)

#### 1.7.5.2 Parametric tests
##### t-tests (paired, unpaired)
##### ANOVA
##### Pearson correlation test
##### Linear regression significance

#### 1.7.5.3 Non-parametric tests
##### Mann-Whitney U test
##### Wilcoxon signed-rank test
##### Kruskal-Wallis test
##### Spearman correlation test

#### 1.7.5.4 Bootstrap & Permutation Testing [A]
##### Bootstrap confidence intervals
##### Permutation tests
##### Resampling distributions

---

## 1.8 Data Leakage & Validation Pitfalls [F] [A]

### 1.8.1 Types of Data Leakage [F] [A]
#### 1.8.1.1 Target Leakage [F]
##### Information from future
##### Training-serving skew
##### Temporal ordering violation
##### Causality reversal

#### 1.8.1.2 Train-Test Contamination [F]
##### Shared preprocessing
##### Feature selection on full data
##### Hyperparameter tuning on test set
##### K-fold pitfalls (scaling fitted on all folds)

#### 1.8.1.3 Label Leakage [A]
##### Derived features containing target info
##### Proxy variables
##### ID-based leakage
##### Grouping artifacts

#### 1.8.1.4 Temporal Leakage [A]
##### Look-ahead bias
##### Survival bias
##### Information published after label date
##### Backtest overfitting

### 1.8.2 Prevention Strategies [F] [A]
#### 1.8.2.1 Pipeline Best Practices [F] [S]
##### Feature engineering after split
##### Preprocessing fitted on train only
##### Validation strategy alignment
##### Reproducible random states

#### 1.8.2.2 Validation-Time Vigilance [A]
##### Sanity checks (performance drops)
##### Correlation with expected variables
##### Unknown feature investigation
##### Performance monitoring post-deployment

---

## 1.9 Distribution Shift & Domain Adaptation [A]

### 1.9.1 Types of Shift [A]
#### 1.9.1.1 Covariate Shift [A]
##### P(X) changes, P(Y|X) constant
##### Importance reweighting
##### Logistic discrimination
##### Domain-adversarial learning

#### 1.9.1.2 Label Shift [A]
##### P(Y) changes, P(X|Y) constant
##### Threshold adjustment
##### Class prior estimation
##### Confusion matrix adaptation

#### 1.9.1.3 Concept Drift [A]
##### Decision boundary shift
##### Prior shift
##### Posterior shift
##### Real vs virtual drift

#### 1.9.1.4 Subpopulation Shift [A]
##### Group performance disparity
##### Fairness concerns
##### Spurious correlations

### 1.9.2 Detection Methods [A]
#### 1.9.2.1 Statistical Tests [A]
##### Kolmogorov-Smirnov test
##### Maximum Mean Discrepancy (MMD)
##### Wasserstein distance
##### Population stability index (PSI)
##### Chi-square test for distributions

#### 1.9.2.2 Model-Based Detection [A]
##### Classifier-based drift detection
##### Density ratio estimation
##### Autoencoders for anomaly
##### Uncertainty quantification

### 1.9.3 Adaptation Strategies [A]
#### 1.9.3.1 Batch Adaptation [A]
##### Batch normalization tuning
##### Self-training
##### Test-time augmentation
##### Feature alignment

#### 1.9.3.2 Continual Learning [A]
##### Incremental learning
##### Catastrophic forgetting prevention
##### Replay buffers
##### Experience replay

#### 1.9.3.3 Robust Learning [A]
##### Group distributionally robust optimization
##### Worst-case group loss
##### Subgroup generalization

---

## 1.10 Causal Inference Basics [A]

### 1.10.1 Causal Fundamentals [A] [T]
#### 1.10.1.1 Causal Graphs [A]
##### Directed acyclic graphs (DAGs)
##### Nodes and edges
##### Confounders
##### Mediators
##### Colliders
##### D-separation criterion

#### 1.10.1.2 Causal Models [A]
##### Structural causal models (SCM)
##### Potential outcomes framework
##### Counterfactuals
##### Treatment effect heterogeneity

### 1.10.2 Identification & Estimation [A]
#### 1.10.2.1 Causal Identification [A]
##### Unconfoundedness assumption
##### Overlap assumption
##### Consistency assumption
##### Backdoor criterion
##### Front-door criterion

#### 1.10.2.2 Estimation Methods [A]
##### Propensity score matching [A]
##### Inverse probability weighting (IPW)
##### Doubly robust estimation
##### Regression adjustment
##### Stratification

### 1.10.3 Treatment Effect Estimation [A]
#### 1.10.3.1 Average Treatment Effect (ATE)
#### 1.10.3.2 Conditional Average Treatment Effect (CATE)
#### 1.10.3.3 Heterogeneous Treatment Effects [A]
##### Causal forests
##### Generalized random forests (GRF)
##### X-learner
##### R-learner
##### Doubly robust learner

---

## 1.11 Experimentation & A/B Testing [F] [S]

### 1.11.1 Experiment Design [F] [S]
#### 1.11.1.1 Test Setup [F]
##### Randomization strategies
###### Complete randomization
###### Stratified randomization
###### Blocked randomization
###### Paired testing

##### Sample size calculation [F]
###### Power and significance level
###### Minimum detectable effect (MDE)
###### Two-sided vs one-sided
###### Type I & II error rates

##### Duration & stopping rules [F]
###### Fixed duration
###### Sequential testing
###### Adaptive sample size

#### 1.11.1.2 Experiment Variants [A]
##### Multi-armed bandits [A]
###### Epsilon-greedy
###### Thompson sampling
###### UCB (Upper Confidence Bound)
###### Regret bounds

##### Contextual bandits [A]
###### Action-dependent features
###### Feature representation
###### Exploration-exploitation

### 1.11.2 Analysis & Metrics [F] [S]
#### 1.11.2.1 Metric Design [F]
##### Primary vs secondary metrics
##### North Star metrics
##### Guardrail metrics
##### Sensitive metrics (early warning)
##### Business metrics vs technical metrics

#### 1.11.2.2 Statistical Analysis [F]
##### Confidence intervals
##### T-tests and p-values
##### Effect size estimation
##### Multiple testing corrections

#### 1.11.2.3 Advanced Analytics [A]
##### CUPED (Controlled experiment using Pre-Experiment Data)
##### Variance reduction techniques
##### Bayesian A/B testing
##### Sequential testing (mSPRT)
##### Mixture effects models

### 1.11.3 Pitfalls & Best Practices [F] [A]
#### 1.11.3.1 Common Mistakes [F] [A]
##### Multiple comparisons bias
##### Optional stopping (peeking)
##### Selection bias
##### Novelty effects
##### Cannibal effect
##### Treatment interference

#### 1.11.3.2 Robustness Checks [A]
##### Balancing checks
##### Placebo tests
##### Mechanism tests
##### Heterogeneity analysis

---

## 1.12 Interpretability & Explainability [F] [A]

### 1.12.1 Model Intrinsic Interpretability [F]
#### 1.12.1.1 Inherently Interpretable Models
##### Linear models (coefficients)
##### Decision trees (path explanation)
##### Rule-based models
##### Generalized additive models (GAMs)
##### Symbolic regression

#### 1.12.1.2 Model Visualization [F]
##### Feature importance plots
##### Partial dependence plots
##### Accumulated local effects (ALE)
##### Individual conditional expectation (ICE)

### 1.12.2 Post-hoc Explanations [F] [A]
#### 1.12.2.1 SHAP Values [F] [A]
##### Shapley value fundamentals [T]
##### Coalition games
##### Marginal contributions
##### TreeSHAP [A]
##### KernelSHAP [A]
##### DeepSHAP [A]
##### SHAP interaction values

#### 1.12.2.2 LIME [F]
##### Local linear approximation
##### Instance perturbation
##### Kernel weighting
##### Feature selection (LIME-G)

#### 1.12.2.3 Other Post-hoc Methods [A]
##### Attention weights (NLP/Vision)
##### Layer-wise relevance propagation (LRP)
##### Integrated gradients
##### Saliency maps
##### Influence functions [A]

### 1.12.3 Fairness & Bias Detection [F] [A]
#### 1.12.3.1 Bias & Fairness Concepts [F] [A]
##### Demographic parity
##### Equalized odds
##### Disparate impact ratio
##### Calibration parity
##### Individual fairness

#### 1.12.3.2 Bias Sources [F] [A]
##### Data collection bias
##### Sampling bias
##### Measurement bias
##### Aggregation bias
##### Evaluation bias
##### Label bias

#### 1.12.3.3 Mitigation Strategies [A]
##### Pre-processing (reweighting, resampling)
##### In-processing (constraint optimization)
##### Post-processing (threshold adjustment)
##### Fairness-aware learning

#### 1.12.3.4 Fairness-Accuracy Trade-offs [A]
##### Pareto frontier
##### Group vs individual fairness
##### Disparate impact vs calibration

---

## 1.13 Robustness & Adversarial ML [A]

### 1.13.1 Adversarial Examples [A]
#### 1.13.1.1 Adversarial Attack Methods [A]
##### FGSM (Fast Gradient Sign Method)
##### PGD (Projected Gradient Descent)
##### C&W (Carlini & Wagner)
##### DeepFool
##### AutoAttack

#### 1.13.1.2 Black-box Attacks [A]
##### Transfer attacks
##### Query-based attacks
##### Score-based gradients
##### Boundary attacks

#### 1.13.1.3 Adversarial Properties [T]
##### Transferability
##### Universal perturbations
##### Physical-world robustness
##### Imperceptibility metrics

### 1.13.2 Adversarial Defense [A]
#### 1.13.2.1 Defense Methods [A]
##### Adversarial training [A]
##### Certified defenses
##### Detection methods
##### Input transformations
##### Model hardening

#### 1.13.2.2 Robustness Evaluation [A]
##### Certified robustness bounds
##### Adversarial perturbation budgets
##### L0, L2, L∞ norms
##### Robustness verification

### 1.13.3 Distribution Robustness [A]
#### 1.13.3.1 Worst-Case Optimization [A]
##### Minimax optimization
##### Distributionally robust optimization (DRO)
##### Uncertainty set specification

#### 1.13.3.2 Robustness to Label Noise [A]
##### Noise types (symmetric, asymmetric)
##### Label smoothing
##### Sample selection
##### Loss correction

---

## 1.14 ML Debugging & Troubleshooting [A] [S]

### 1.14.1 Diagnosis Framework [A] [S]
#### 1.14.1.1 Performance Gap Analysis [A]
##### Training error vs test error
##### Learning curves interpretation
##### Bias vs variance identification
##### Model-centric vs data-centric view

#### 1.14.1.2 Common Issues & Solutions [A]
##### High bias (underfitting)
###### Larger model capacity
###### More features
###### Better feature engineering
###### Reduce regularization

##### High variance (overfitting)
###### More training data
###### Reduce model complexity
###### Increase regularization
###### Data augmentation

##### Poor feature representation
###### Exploratory data analysis
###### Feature statistics
###### Feature correlations
###### Outlier analysis

### 1.14.2 Debugging Tools & Techniques [A]
#### 1.14.2.1 Error Analysis [A]
##### Confusion matrix analysis
##### False positive vs false negative breakdown
##### Hard example identification
##### Calibration curves
##### Residual analysis

#### 1.14.2.2 Ablation Studies [A]
##### Feature removal impact
##### Component removal
##### Architectural changes
##### Training procedure changes

#### 1.14.2.3 Model Surgery [A]
##### Unit dissection
##### Representation analysis
##### Embedding space analysis

---

# 2. Deep Learning

## 2.1 Fundamentals & Neural Network Basics

### 2.1.1 Perceptrons & Neural Networks [F] [T]
#### 2.1.1.1 Single Layer Perceptron [T]
##### Threshold logic
##### Linear separability
##### XOR problem
##### Weight updates
##### Rosenblatt's theorem

#### 2.1.1.2 Multi-Layer Perceptrons (MLPs) [F]
##### Hidden layers
##### Activation functions
##### Backpropagation
##### Universal approximation theorem
##### Network depth vs width

#### 2.1.1.3 Network Representation [T]
##### Neurons as functions
##### Layer composition
##### Computational graphs
##### Forward pass mechanics
##### Backward pass mechanics

### 2.1.2 Activation Functions [F]
#### 2.1.2.1 Traditional Activations [F]
##### Sigmoid [F]
###### Squashing property
###### Gradient issues (vanishing gradients)
###### Output interpretation as probability

##### Tanh [F]
###### Symmetry around origin
###### Stronger gradients than sigmoid
###### Output range [-1, 1]

##### ReLU [F] [A]
###### Rectified linear unit
###### Sparsity-inducing
###### Dying ReLU problem
###### Computational efficiency

#### 2.1.2.2 ReLU Variants [A]
##### Leaky ReLU
##### ELU (Exponential Linear Unit)
##### SELU (Scaled ELU) [A]
##### GELU (Gaussian Error Linear Unit)
##### Mish [A]
##### SiLU/Swish [A]
##### GLU variants [A]

#### 2.1.2.3 Advanced Activations [A]
##### Multiplicative interactions
##### Non-monotonic activations
##### Parametric activations
##### Context-dependent activations

### 2.1.3 Backpropagation & Automatic Differentiation [F] [T]
#### 2.1.3.1 Computational Graphs [F] [T]
##### Forward propagation
##### Backward propagation
##### Chain rule application
##### Topological ordering
##### Dynamic vs static graphs

#### 2.1.3.2 Gradient Computation [F]
##### Jacobian matrices
##### Hessian matrices
##### Gradient accumulation
##### Numerical gradient checking

#### 2.1.3.3 Backpropagation Variants [A]
##### Truncated BPTT (time series)
##### Reverse-mode differentiation
##### Forward-mode differentiation
##### Mixed-mode differentiation

#### 2.1.3.4 Automatic Differentiation Frameworks [S]
##### TensorFlow computational graphs
##### PyTorch dynamic computation graphs
##### JAX functional transformations
##### Framework comparison

---

## 2.2 Convolutional Neural Networks (CNNs)

### 2.2.1 Convolution Operations [F]
#### 2.2.1.1 Convolution Fundamentals [F]
##### Convolution operation
##### Kernels / filters
##### Padding strategies
###### Valid (no padding)
###### Same (output size preserves)
###### Full padding

##### Stride
##### Receptive field
##### Parameter sharing
##### Spatial invariance

#### 2.2.1.2 Convolution Variants [A]
##### 1D convolution (time series)
##### 2D convolution (images)
##### 3D convolution (video)
##### Grouped convolution
##### Depthwise convolution
##### Pointwise convolution
##### Separable convolution [A]
##### Dilated convolution (atrous) [A]
##### Deformable convolution [A]

#### 2.2.1.3 Pooling Operations [F]
##### Max pooling
##### Average pooling
##### Stochastic pooling
##### Spatial pyramid pooling (SPP)
##### Adaptive pooling

### 2.2.2 Classic CNN Architectures [F]
#### 2.2.2.1 Pioneering Models [F]
##### LeNet [T]
##### AlexNet [F]
##### VGGNet [F]
##### Architectural choices
##### Interpretation

#### 2.2.2.2 Modern Efficient Architectures [A]
##### MobileNets [A]
###### Depthwise separable convolutions
###### Width multiplier
###### Resolution multiplier

##### EfficientNet [A]
###### Compound scaling (depth, width, resolution)
###### AutoML architecture search
###### Mobile vs server variants

##### ShuffleNet [A]
##### SqueezeNet [A]
##### MnasNet [A]

### 2.2.3 Residual Networks & Skip Connections [F] [A]
#### 2.2.3.1 ResNets [F] [A]
##### Residual blocks
##### Identity shortcuts
##### Bottleneck blocks
##### Skip connection mathematics
##### Gradient flow benefits

#### 2.2.3.2 Skip Connection Variants [A]
##### Dense connections (DenseNet)
##### Inception modules (GoogLeNet)
##### Squeeze-and-excitation blocks (SE-ResNet)
##### Channel attention
##### Spatial attention

### 2.2.4 Advanced CNN Concepts [A]
#### 2.2.4.1 Multi-Scale Processing [A]
##### Feature pyramids
##### ASPP (Atrous Spatial Pyramid Pooling)
##### FPN (Feature Pyramid Networks)
##### Multi-scale training

#### 2.2.4.2 Normalization Techniques [A]
##### Batch normalization
##### Group normalization
##### Instance normalization
##### Layer normalization
##### Whitening approaches

#### 2.2.4.3 CNN Interpretability [A]
##### Visualization of filters
##### Feature maps
##### Saliency maps
##### Class activation maps (CAM)
##### Grad-CAM
##### Attention-based explanations

---

## 2.3 Recurrent Neural Networks (RNNs)

### 2.3.1 RNN Fundamentals [F]
#### 2.3.1.1 Basic RNN Architecture [F]
##### Recurrent connections
##### Hidden states
##### Unfolding in time
##### Parameter sharing across time
##### Elman networks
##### Jordan networks

#### 2.3.1.2 RNN Training [F]
##### Backpropagation Through Time (BPTT)
##### Truncated BPTT
##### Teacher forcing
##### Output feedback

#### 2.3.1.3 RNN Challenges [F] [A]
##### Vanishing gradients [F]
##### Exploding gradients [F]
##### Gradient clipping
##### Initialization strategies

### 2.3.2 LSTM & GRU [F]
#### 2.3.2.1 Long Short-Term Memory (LSTM) [F]
##### Memory cells
##### Input gate
##### Forget gate
##### Output gate
##### Cell state
##### Peephole connections [A]
##### Coupled input-forget gates [A]

#### 2.3.2.2 Gated Recurrent Unit (GRU) [F]
##### Reset gate
##### Update gate
##### Simplified LSTM
##### GRU vs LSTM trade-off

#### 2.3.2.3 LSTM Extensions [A]
##### Bidirectional LSTM
##### Deep LSTM (stacked)
##### Attention-augmented LSTM
##### Recurrent batch normalization
##### Layer normalization in LSTM

### 2.3.3 Sequence Modeling [F] [A]
#### 2.3.3.1 Sequence-to-Sequence (Seq2Seq) [F] [A]
##### Encoder-decoder architecture
##### Teacher forcing
##### Beam search
##### Attention mechanisms in seq2seq
##### Bucketing (variable length)
##### Scheduled sampling

#### 2.3.3.2 Advanced RNN Variants [A]
##### Clockwork RNNs
##### Hierarchical RNNs
##### Bidirectional encoder representations
##### Residual RNNs
##### Recurrent dropout [A]

---

## 2.4 Attention Mechanisms [F] [A]

### 2.4.1 Attention Fundamentals [F] [A]
#### 2.4.1.1 Attention Concept [F] [A]
##### Query, Key, Value framework
##### Attention weights
##### Weighted sum
##### Context vector
##### Alignment mechanism

#### 2.4.1.2 Attention Types [F] [A]
##### Additive attention (Bahdanau) [F] [A]
##### Multiplicative attention (Luong) [F] [A]
##### Dot-product attention
##### Scaled dot-product attention [A]
##### General linear attention
##### Bilinear attention

#### 2.4.1.3 Attention Variants [A]
##### Self-attention
##### Cross-attention
##### Multi-head attention [A]
##### Multi-hop attention

### 2.4.2 Multi-Head Attention [A]
#### 2.4.2.1 Mechanics [A]
##### Multiple representation subspaces
##### Head dimension
##### Concatenation
##### Linear projection
##### Information diversity

#### 2.4.2.2 Multi-Head Analysis [A]
##### Head specialization [A]
##### Attention pattern redundancy
##### Head pruning
##### Head interpolation

### 2.4.3 Advanced Attention [A]
#### 2.4.3.1 Efficient Attention [A]
##### Sparse attention
##### Local attention windows
##### Strided attention
##### Fixed patterns (Longformer, BigBird)
##### Learned patterns

#### 2.4.3.2 Linear Complexity Attention [A]
##### Kernel-based attention
##### Performer
##### Linear transformers
##### Approximation trade-offs

#### 2.4.3.3 Relative Position Encoding [A]
##### Shaw's method
##### Relative position bias
##### Rotary position embeddings (RoPE) [A]
##### ALiBi (Attention with Linear Biases) [A]

---

## 2.5 Transformer Architecture [F] [A]

### 2.5.1 Transformer Fundamentals [F] [A]
#### 2.5.1.1 Core Components [F] [A]
##### Encoder block
##### Decoder block
##### Positional encoding [F] [A]
###### Sinusoidal encoding
###### Learned encodings
###### Rotary embeddings (RoPE)
###### Alibi position biases

##### Token embeddings
##### Embedding + positional encoding

#### 2.5.1.2 Encoder-Decoder Structure [F] [A]
##### Self-attention in encoder
##### Cross-attention in decoder
##### Autoregressive decoding
##### Attention masking (causal)

#### 2.5.1.3 Transformer Mechanics [F] [A]
##### Layer normalization placement
##### Feed-forward networks (FFN)
##### Residual connections
##### Output projection
##### Softmax normalization

### 2.5.2 Transformer Variants [A]
#### 2.5.2.1 Encoder-Only [A]
##### BERT architecture [A]
##### RoBERTa improvements
##### ALBERT (A Lite BERT)
##### DeBERTa (Decoding-enhanced BERT)
##### Bidirectional encoders

#### 2.5.2.2 Decoder-Only [A]
##### GPT architecture [A]
##### Autoregressive language modeling
##### Causal self-attention
##### GPT variants (GPT-2, GPT-3, GPT-Neo)

#### 2.5.2.3 Encoder-Decoder [A]
##### T5 architecture
##### Prefix tuning
##### ByT5 (byte-level)
##### mT5 (multilingual T5)

#### 2.5.2.4 Hybrid Architectures [A]
##### PEGASUS (pre-training with extracted GAP sentences)
##### Hybrid attention mechanisms
##### Unified architectures

### 2.5.3 Transformer Optimization [A]
#### 2.5.3.1 Computational Efficiency [A]
##### Quadratic complexity of attention
##### Linear attention approximations
##### Knowledge distillation for transformers
##### Pruning techniques
##### Quantization strategies [A]

#### 2.5.3.2 Memory Efficiency [A]
##### Gradient checkpointing
##### Flash Attention [A]
##### Memory-efficient attention
##### Activation functions memory
##### Parameter sharing

#### 2.5.3.3 Scaling Strategies [A]
##### Model parallelism
##### Data parallelism
##### Pipeline parallelism
##### Tensor parallelism
##### Distributed training frameworks

---

## 2.6 Generative Models

### 2.6.1 Autoencoders [F]
#### 2.6.1.1 Standard Autoencoders [F]
##### Encoder network
##### Latent representation
##### Decoder network
##### Reconstruction loss
##### Undercomplete vs overcomplete
##### Sparse autoencoders

#### 2.6.1.2 Variational Autoencoders (VAE) [A]
##### Latent variable model
##### ELBO (Evidence Lower Bound) [T]
##### KL divergence term
##### Reconstruction term
##### Reparameterization trick [T]
##### Posterior collapse problem [A]
##### VAE variants (beta-VAE, etc.)

#### 2.6.1.3 Denoising Autoencoders [A]
##### Noise injection
##### Denoising objective
##### Score matching [T]
##### Connection to diffusion models

### 2.6.2 Generative Adversarial Networks (GANs) [F] [A]
#### 2.6.2.1 GAN Fundamentals [F] [A]
##### Generator network
##### Discriminator network
##### Adversarial training
##### Min-max game
##### Nash equilibrium
##### Mode collapse [A]

#### 2.6.2.2 GAN Variants [A]
##### DCGAN (Deep Convolutional)
##### WGAN (Wasserstein GAN)
##### Spectral normalization [A]
##### Gradient penalty (WGAN-GP)
##### Progressive GANs (ProGAN) [A]
##### StyleGAN [A]
##### Conditional GANs (cGAN)

#### 2.6.2.3 GAN Training Challenges [A]
##### Mode collapse
##### Training instability
##### Hyperparameter sensitivity
##### Evaluation difficulties
##### Inception Score
##### Fréchet Inception Distance (FID)

### 2.6.3 Normalizing Flows [T] [A]
#### 2.6.3.1 Flow Fundamentals [T]
##### Change of variables formula
##### Invertible transformations
##### Determinant of Jacobian
##### Coupling layers [A]
##### Masked autoencoder flows [A]

#### 2.6.3.2 Flow Models [A]
##### Glow
##### Flow++
##### Coupling-based flows
##### Autoregressive flows

### 2.6.4 Energy-Based Models [T] [A]
#### 2.6.4.1 Energy Function Perspective [T]
##### Boltzmann machines
##### Restricted Boltzmann machines (RBM)
##### Energy landscape
##### Gibbs sampling
##### Contrastive divergence [T]

#### 2.6.4.2 Score-Based Models [A]
##### Score function (gradient of log probability)
##### Score matching [T]
##### Denoising score matching
##### Connection to diffusion

### 2.6.5 Diffusion Models [A]
#### 2.6.5.1 Diffusion Process [A]
##### Forward diffusion process
##### Noise schedule
##### Reverse process
##### Denoising diffusion probabilistic models (DDPM)
##### Variance schedule (linear, cosine, exponential)

#### 2.6.5.2 DDPM Training & Sampling [A]
##### Training objective (noise prediction)
##### Sampling (ancestral sampling)
##### Deterministic sampling (DDIM)
##### Sampling speed improvements
##### Classifier guidance [A]

#### 2.6.5.3 Diffusion Variants [A]
##### Latent diffusion models [A]
##### Continuous diffusion (score-based)
##### Flow matching [A]
##### Consistency models [A]

---

## 2.7 Probabilistic Models & Bayesian Deep Learning

### 2.7.1 Bayesian Neural Networks [A]
#### 2.7.1.1 Weight Uncertainty [A]
##### Prior distribution over weights
##### Posterior inference
##### Variational inference [A]
##### MCMC sampling
##### Laplace approximation

#### 2.7.1.2 Approximate Inference [A]
##### Mean-field variational inference
##### Expectation propagation
##### Variational dropout [A]
##### Concrete dropout [A]

#### 2.7.1.3 Uncertainty Quantification [A]
##### Aleatoric uncertainty
##### Epistemic uncertainty
##### Calibration of uncertainty
##### Confidence intervals

### 2.7.2 Probabilistic Inference [T] [A]
#### 2.7.2.1 Variational Inference [T]
##### Variational lower bound
##### ELBO optimization
##### Amortized inference
##### Reparameterization trick

#### 2.7.2.2 MCMC in Deep Learning [A]
##### Hamiltonian Monte Carlo (HMC)
##### Stochastic gradient Langevin dynamics (SGLD)
##### Posterior sampling
##### Mixing time

---

## 2.8 Self-Supervised & Contrastive Learning [A]

### 2.8.1 Contrastive Learning [A]
#### 2.8.1.1 Contrastive Loss Functions [A]
##### Triplet loss [A]
##### Margin-based losses
##### NT-Xent (normalized temperature-scaled cross entropy)
##### InfoNCE loss
##### Contrastive divergence

#### 2.8.1.2 Contrastive Methods [A]
##### SimCLR [A]
###### Data augmentation
###### Positive pairs
###### Contrastive loss
###### Projection head

##### MoCo (Momentum Contrast) [A]
###### Memory bank
###### Momentum encoder
###### Queue mechanism

##### BYOL (Bootstrap Your Own Latent) [A]
###### Predictor network
###### Exponential moving average (EMA)
###### No negative pairs needed

##### SwAV [A]
##### Dino [A]

### 2.8.2 Non-Contrastive Self-Supervised [A]
#### 2.8.2.1 Methods [A]
##### Masked language modeling (MLM) [A]
##### Masked image modeling [A]
##### Rotation prediction
##### Jigsaw puzzles
##### Context encoder

#### 2.8.2.2 Augmentation-Invariant Learning [A]
##### Invariance vs equivariance
##### Sufficient statistics
##### Whitening & decorrelation

---

## 2.9 Meta-Learning & Few-Shot Learning [A]

### 2.9.1 Meta-Learning Concepts [A]
#### 2.9.1.1 Learning to Learn [A]
##### Task distribution
##### Meta-train vs meta-test
##### Meta-objectives
##### Inner vs outer loop

#### 2.9.1.2 Optimization-Based Meta-Learning [A]
##### Model-Agnostic Meta-Learning (MAML) [A]
##### Bi-level optimization
##### Inner gradient steps
##### Second-order information

#### 2.9.1.3 Metric-Based Meta-Learning [A]
##### Siamese networks
##### Prototypical networks
##### Matching networks
##### Relation networks

### 2.9.2 Few-Shot Learning [A]
#### 2.9.2.1 Problem Formulation [A]
##### N-way, K-shot setup
##### Episodic training
##### Task diversity

#### 2.9.2.2 Few-Shot Methods [A]
##### Prototypical networks [A]
##### Matching networks [A]
##### Relation networks [A]
##### Transductive few-shot [A]

---

## 2.10 Neural Architecture Search (NAS) [A]

### 2.10.1 Search Strategies [A]
#### 2.10.1.1 Search Space [A]
##### Micro vs macro search
##### Modular building blocks
##### Operations vocabulary
##### Connection patterns

#### 2.10.1.2 Search Methods [A]
##### Grid search
##### Random search
##### Evolutionary algorithms
##### Reinforcement learning-based NAS
##### Gradient-based NAS [A]
##### Bayesian optimization

#### 2.10.1.3 Efficiency [A]
##### Early stopping
##### Performance prediction
##### Weight sharing
##### One-shot NAS [A]
##### Zero-cost proxies [A]

### 2.10.2 Benchmark Datasets [A]
##### ImageNet
##### CIFAR-10/100
##### Fashion-MNIST

---

## 2.11 Knowledge Distillation [A]

### 2.11.1 Distillation Concepts [A]
#### 2.11.1.1 Teacher-Student Framework [A]
##### Large teacher model
##### Small student model
##### Knowledge transfer
##### Temperature scaling
##### Soft targets

#### 2.11.1.2 Distillation Techniques [A]
##### Response-based distillation
##### Feature-based distillation
##### Relation-based distillation
##### Dark knowledge

#### 2.11.1.3 Advanced Distillation [A]
##### Multi-teacher distillation
##### Self-distillation
##### Online distillation
##### Mutual learning

---

## 2.12 Model Compression [A] [S]

### 2.12.1 Pruning [A] [S]
#### 2.12.1.1 Structured Pruning [A]
##### Channel pruning
##### Filter pruning
##### Layer pruning
##### Block pruning
##### Lottery ticket hypothesis [A]

#### 2.12.1.2 Unstructured Pruning [A]
##### Weight pruning
##### Magnitude-based pruning
##### Gradual pruning
##### Sensitivity analysis

### 2.12.2 Quantization [A] [S]
#### 2.12.2.1 Quantization Basics [A]
##### Integer quantization
##### Fixed-point arithmetic
##### Bit-width selection
##### Symmetric vs asymmetric

#### 2.12.2.2 Quantization Methods [A]
##### Post-training quantization [A]
##### Quantization-aware training (QAT) [A]
##### Mixed-precision quantization [A]
##### Knowledge distillation for quantization

#### 2.12.2.3 Quantization Techniques [A]
##### Min-max quantization
##### KL-divergence based quantization
##### Percentile-based quantization

### 2.12.3 Low-Rank Decomposition [A]
##### Matrix factorization
##### Tensor decomposition
##### SVD-based compression

---

# 3. Statistics & Mathematics

## 3.1 Probability Theory [F] [T]

### 3.1.1 Fundamentals [F] [T]
#### 3.1.1.1 Probability Axioms [F] [T]
##### Sample spaces
##### Events
##### Probability measures
##### Kolmogorov axioms
##### Sigma algebras

#### 3.1.1.2 Conditional Probability [F] [T]
##### Bayes' theorem [F]
##### Independence
##### Chain rule
##### Law of total probability
##### Prior, likelihood, posterior

#### 3.1.1.3 Random Variables [F] [T]
##### Discrete distributions
##### Continuous distributions
##### PMF (probability mass function)
##### PDF (probability density function)
##### CDF (cumulative distribution function)
##### Quantile function

### 3.1.2 Distributions [F]
#### 3.1.2.1 Discrete Distributions [F]
##### Bernoulli
##### Binomial
##### Categorical / Multinomial
##### Poisson
##### Geometric
##### Hypergeometric

#### 3.1.2.2 Continuous Distributions [F]
##### Uniform
##### Normal (Gaussian) [F]
##### Exponential
##### Gamma
##### Beta
##### Chi-square
##### Student's t
##### Laplace
##### Cauchy (pathological examples)

#### 3.1.2.3 Multivariate Distributions [F]
##### Multivariate Normal [F]
##### Covariance matrix properties
##### Conditional distributions
##### Marginal distributions
##### Wishart distribution [T]
##### Dirichlet distribution [T]

### 3.1.3 Expectation & Moments [F] [T]
#### 3.1.3.1 Expected Value [F]
##### Linearity of expectation
##### Law of the unconscious statistician
##### Indicator random variables

#### 3.1.3.2 Variance & Covariance [F]
##### Variance definition
##### Variance formulas
##### Covariance definition
##### Correlation coefficient
##### Covariance matrix
##### Positive semi-definiteness

#### 3.1.3.3 Higher Moments [A]
##### Skewness
##### Kurtosis
##### Moment generating functions
##### Characteristic functions [T]

### 3.1.4 Concentration Inequalities [T] [A]
#### 3.1.4.1 Bounds [T]
##### Markov's inequality
##### Chebyshev's inequality
##### Chernoff bound
##### Hoeffding's inequality
##### Bernstein's inequality

#### 3.1.4.2 Applications [A]
##### Sample complexity bounds
##### Generalization bounds
##### PAC learning

---

## 3.2 Statistical Inference [F] [T]

### 3.2.1 Estimation [F]
#### 3.2.1.1 Point Estimation [F]
##### Estimators (unbiased, consistent, efficient)
##### Maximum Likelihood Estimation (MLE) [F]
###### Likelihood function
###### Log-likelihood
###### Score function
###### Fisher information

##### Method of Moments [T]
##### Bayesian Estimation [A]
###### MAP (Maximum A Posteriori)
###### Posterior distribution
###### Credible intervals

#### 3.2.1.2 Interval Estimation [F]
##### Confidence intervals [F]
##### Coverage probability
##### Confidence level
##### Bootstrap confidence intervals

### 3.2.2 Hypothesis Testing [F]
#### 3.2.2.1 Test Formulation [F]
##### Null and alternative hypotheses
##### Type I error (false positive)
##### Type II error (false negative)
##### Power of test
##### Test statistic
##### Rejection region

#### 3.2.2.2 Common Tests [F]
##### Z-test
##### T-test (one-sample, two-sample, paired)
##### Chi-square test
##### Kolmogorov-Smirnov test
##### Anderson-Darling test
##### Shapiro-Wilk test (normality)

#### 3.2.2.3 P-values & Significance [F]
##### P-value definition
##### Significance level
##### Multiple testing corrections [A]
###### Bonferroni correction
###### FDR (False Discovery Rate)
###### Benjamini-Hochberg procedure

#### 3.2.2.4 Advanced Testing [A]
##### Equivalence testing
##### Non-inferiority testing
##### Sequential hypothesis testing

---

## 3.3 Bayesian Statistics [A] [T]

### 3.3.1 Bayesian Framework [A] [T]
#### 3.3.1.1 Bayes' Theorem [F] [T]
##### Prior beliefs
##### Likelihood
##### Evidence (marginal likelihood)
##### Posterior distribution
##### Conjugate priors

#### 3.3.1.2 Prior Specification [A]
##### Informative priors
##### Weakly informative priors
##### Non-informative priors
##### Improper priors
##### Hyperpriors

#### 3.3.1.3 Posterior Inference [A]
##### Point estimates (posterior mean, MAP, median)
##### Credible intervals
##### Posterior predictive distribution
##### Predictive checks

### 3.3.2 Bayesian Hierarchical Models [A]
#### 3.3.2.1 Hierarchical Structure [A]
##### Parameters as random variables
##### Partial pooling
##### Shrinkage estimation
##### Exchangeability

#### 3.3.2.2 Applications [A]
##### Meta-analysis
##### Multi-site studies
##### Sparse estimation

### 3.3.3 Approximation Methods [A] [T]
#### 3.3.3.1 Variational Inference [A] [T]
##### Variational lower bound (ELBO)
##### Mean-field approximation
##### Amortized inference
##### Black-box variational inference

#### 3.3.3.2 Markov Chain Monte Carlo [A] [T]
##### Metropolis-Hastings algorithm
##### Gibbs sampling
##### Hamiltonian Monte Carlo (HMC)
##### Convergence diagnosis
##### Effective sample size
##### Burn-in period

---

## 3.4 Decision Theory & Information Theory

### 3.4.1 Decision Theory [T]
#### 3.4.1.1 Loss Functions [F] [T]
##### 0-1 loss
##### Squared loss
##### Absolute loss
##### Quantile loss
##### Cross-entropy loss [F]
##### Custom domain losses

#### 3.4.1.2 Risk & Bayes Risk [T]
##### Expected loss
##### Empirical risk
##### Bayes optimal classifier
##### Minimax risk
##### Regret bounds

### 3.4.2 Information Theory [T] [A]
#### 3.4.2.1 Entropy [T]
##### Shannon entropy
##### Differential entropy
##### Joint entropy
##### Conditional entropy
##### Mutual information [T]

#### 3.4.2.2 Divergences [T] [A]
##### KL divergence [F] [T]
##### JS divergence (symmetric KL)
##### Hellinger distance
##### Wasserstein distance [A]
##### Earth Mover's Distance

#### 3.4.2.3 Information-Theoretic Learning [A]
##### Minimum description length (MDL)
##### Maximum entropy
##### Information bottleneck [A]
##### Rate-distortion theory

---

## 3.5 Linear Algebra [F] [T]

### 3.5.1 Matrices & Vectors [F]
#### 3.5.1.1 Matrix Operations [F]
##### Transpose
##### Determinant
##### Trace
##### Rank
##### Null space & column space
##### Matrix norms (Frobenius, spectral, nuclear)

#### 3.5.1.2 Matrix Decompositions [F]
##### Eigendecomposition [F]
##### Singular Value Decomposition (SVD) [F]
###### Left/right singular vectors
###### Singular values
###### Rank approximation
###### Truncated SVD

##### QR decomposition
##### Cholesky decomposition
##### LU decomposition
##### Schur decomposition [T]

#### 3.5.1.3 Special Matrices [F]
##### Symmetric matrices
##### Positive definite matrices
##### Orthogonal matrices
##### Projection matrices
##### Idempotent matrices

### 3.5.2 Vector Spaces & Norms [F]
#### 3.5.2.1 Vector Spaces [F] [T]
##### Basis
##### Dimension
##### Span
##### Orthogonality
##### Orthonormal basis

#### 3.5.2.2 Norms [F]
##### L0 norm (sparsity)
##### L1 norm (Manhattan)
##### L2 norm (Euclidean)
##### Lp norms
##### Matrix norms
##### Dual norms

### 3.5.3 Advanced Linear Algebra [A]
#### 3.5.3.1 Matrix Calculus [A]
##### Gradient of scalar w.r.t. vector
##### Gradient of scalar w.r.t. matrix
##### Jacobian matrices
##### Hessian matrices
##### Vector calculus identities

#### 3.5.3.2 Optimization Perspectives [A]
##### Quadratic forms
##### Positive definiteness in optimization
##### Convex sets & functions
##### Convex optimization [A]

---

## 3.6 Calculus & Optimization [F] [T]

### 3.6.1 Differential Calculus [F]
#### 3.6.1.1 Derivatives [F]
##### Univariate derivatives
##### Partial derivatives
##### Directional derivatives
##### Gradient vector
##### Jacobian
##### Hessian

#### 3.6.1.2 Taylor Expansion [F]
##### First-order Taylor approximation
##### Second-order approximation
##### Multivariate Taylor expansion
##### Approximation quality

### 3.6.2 Integral Calculus [T]
#### 3.6.2.1 Integrals [T]
##### Riemann integrals
##### Lebesgue integrals
##### Improper integrals
##### Numerical integration

#### 3.6.2.2 Probability Integration [T]
##### Integrating probability densities
##### Normalizing constant computation
##### Expectation as integral

### 3.6.3 Convex Analysis [A] [T]
#### 3.6.3.1 Convexity [A] [T]
##### Convex sets
##### Convex functions
##### Jensen's inequality
##### Epigraph
##### Convex hull

#### 3.6.3.2 Optimization Conditions [A] [T]
##### First-order condition (gradient = 0)
##### Second-order condition (Hessian positive definite)
##### KKT conditions
##### Complementary slackness
##### Constraint qualification

#### 3.6.3.3 Duality [A] [T]
##### Lagrangian function
##### Dual problem
##### Weak duality
##### Strong duality (Slater's condition)
##### Dual decomposition

---

## 3.7 Stochastic Processes [A] [T]

### 3.7.1 Basic Concepts [A]
#### 3.7.1.1 Definition & Classification [A]
##### Stochastic processes
##### Markov property
##### Markov chains
##### Stationary processes
##### Ergodicity

#### 3.7.1.2 Markov Chains [A]
##### Transition matrix
##### Steady-state distribution
##### Irreducibility
##### Aperiodicity
##### Detailed balance

### 3.7.2 Continuous-Time Processes [A]
#### 3.7.2.1 Brownian Motion [A]
##### Wiener process
##### Properties (continuity, no differentiability)
##### Fractional Brownian motion
##### Geometric Brownian motion

#### 3.7.2.2 Poisson Processes [A]
##### Arrival times
##### Intensity function
##### Non-homogeneous Poisson
##### Compound Poisson process

---

# 4. Natural Language Processing (NLP)

## 4.1 Fundamentals & Text Preprocessing

### 4.1.1 Text Representations [F]
#### 4.1.1.1 Character & Token Level [F]
##### Character encoding
##### Byte-pair encoding (BPE) [F] [A]
##### WordPiece tokenization [A]
##### SentencePiece [A]
##### Subword tokenization motivation

#### 4.1.1.2 Word Representations [F]
##### One-hot encoding
##### Bag of words
##### TF-IDF [F]
##### Sparse vs dense representations
##### Distributional semantics

### 4.1.2 Text Preprocessing [F]
#### 4.1.2.1 Normalization [F]
##### Lowercasing
##### Punctuation removal
##### Stemming (Porter stemmer)
##### Lemmatization (morphological analysis)
##### Accent removal

#### 4.1.2.2 Tokenization [F]
##### Word tokenization
##### Sentence tokenization
##### Sub-word tokenization
##### Special token handling

#### 4.1.2.3 Advanced Preprocessing [A]
##### Stop word removal (context-dependent)
##### Spell correction
##### HTML/XML parsing
##### Emoji handling

### 4.1.3 Linguistic Structure [F]
#### 4.1.3.1 Morphology [F]
##### Morphemes
##### Inflection vs derivation
##### Compound words
##### Irregular forms

#### 4.1.3.2 Syntax [F]
##### Parts-of-speech (POS) tagging
##### Dependency parsing
##### Constituency parsing
##### Treebanks
##### Parse trees

#### 4.1.3.3 Semantics [F]
##### Semantic roles
##### Word sense
##### Polysemy
##### Named entity recognition (NER)

#### 4.1.3.4 Discourse [A]
##### Coreference resolution
##### Ellipsis
##### Anaphora
##### Discourse relations
##### Rhetorical structure

---

## 4.2 Word Embeddings

### 4.2.1 Classical Embeddings [F]
#### 4.2.1.1 Word2Vec [F]
##### Skip-gram architecture
##### CBOW (Continuous Bag of Words)
##### Negative sampling [F]
##### Hierarchical softmax
##### Word analogy properties
##### Limitations (static, sense-agnostic)

#### 4.2.1.2 GloVe [F]
##### Global co-occurrence statistics
##### Weighted least squares objective
##### Context windows
##### Bias terms
##### Combination with Word2Vec benefits

#### 4.2.1.3 FastText [F]
##### Subword information
##### Character n-grams
##### Out-of-vocabulary handling
##### Multilingual embeddings

### 4.2.2 Contextualized Embeddings [F] [A]
#### 4.2.2.1 Contextual Representations [A]
##### Layer-wise representation
##### Token-level context sensitivity
##### ELMo [A]
###### BiLSTM layers
###### Weighted sum of layers
###### Task-specific adaptation

#### 4.2.2.2 Transformer-Based Embeddings [A]
##### BERT representations [A]
##### RoBERTa improvements
##### ALBERT parameter sharing
##### DeBERTa disentangled attention
##### XLNet permutation language modeling
##### ELECTRA discriminative pretraining

### 4.2.3 Sentence Embeddings [A]
#### 4.2.3.1 Methods [A]
##### Average word embeddings
##### Doc2Vec / Paragraph vectors
##### Universal Sentence Encoder
##### Sentence-BERT (SBERT)
##### Sentence transformers (contrastive learning)

#### 4.2.3.2 Evaluation [A]
##### Semantic Textual Similarity (STS) tasks
##### Downstream task performance
##### Alignment properties

---

## 4.3 Language Models

### 4.3.1 Autoregressive Language Models [F]
#### 4.3.1.1 N-gram Models [F]
##### Markov assumption
##### Count-based probabilities
##### Smoothing techniques
###### Laplace smoothing
###### Good-Turing smoothing
###### Kneser-Ney smoothing
###### Interpolation & backoff

#### 4.3.1.2 Neural Autoregressive Models [F]
##### RNN-based language models
##### LSTM language models
##### GRU language models
##### Softmax output layer
##### Perplexity metric [F]

#### 4.3.1.3 Transformer Language Models [F] [A]
##### GPT architecture [A]
##### Causal self-attention [A]
##### Token-by-token generation
##### Autoregressive decoding
##### Teacher forcing vs scheduled sampling

### 4.3.2 Bidirectional / Masked Language Models [A]
#### 4.3.2.1 Masked Language Modeling [A]
##### BERT's MLM objective [A]
##### Random masking strategy
##### Special token handling [MASK], [CLS], [SEP]
##### Subword masking variations

#### 4.3.2.2 Permutation Language Modeling [A]
##### XLNet permutation objective
##### Two-stream attention
##### Relative position biases

### 4.3.3 Encoder-Decoder Language Models [A]
#### 4.3.3.1 Seq2Seq Pretraining [A]
##### T5 pretraining objectives
##### Span corruption
##### Denoising pretraining
##### Encoder-decoder alignment

#### 4.3.3.2 Multi-Task Pretraining [A]
##### ELECTRA (discriminator training)
##### UNILM (unified language modeling)
##### ERNIE (entity-aware pretraining)

---

## 4.4 Sequence Labeling & Structured Prediction

### 4.4.1 Sequence Tagging [F]
#### 4.4.1.1 Tasks [F]
##### Part-of-speech tagging
##### Named entity recognition (NER)
##### Chunking
##### Slot filling

#### 4.4.1.2 Models [F]
##### BiLSTM-CRF [F] [A]
###### BiLSTM encoder
###### CRF decoder
###### Viterbi decoding
###### Transition scores

##### Transformer-based tagging
##### Softmax-only tagging

### 4.4.2 Conditional Random Fields (CRF) [A]
#### 4.4.2.1 CRF Theory [A]
##### Undirected graphical models
##### Factor graphs
##### Clique potentials
##### Partition function
##### Log-linear models

#### 4.4.2.2 CRF Training & Inference [A]
##### Maximum likelihood training
##### Inference (Viterbi, forward-backward)
##### Beam search decoding
##### Marginal probabilities

### 4.4.3 Parsing [A]
#### 4.4.3.1 Dependency Parsing [A]
##### Head selection for each token
##### Transition-based parsing
##### Graph-based parsing
##### Neural dependency parsers

#### 4.4.3.2 Constituency Parsing [A]
##### Span-based parsing
##### Shift-reduce parsers
##### PCFG (Probabilistic CFG)
##### Neural constituency parsing

---

## 4.5 Machine Translation [A]

### 4.5.1 Translation Models [A]
#### 4.5.1.1 Statistical Machine Translation [T]
##### Phrase-based translation
##### Alignment models (IBM models)
##### Language model + translation model
##### Reordering models

#### 4.5.1.2 Neural Machine Translation [A]
##### Encoder-decoder with attention [A]
##### Transformer-based NMT [A]
##### Multilingual NMT
##### Back-translation for data augmentation

#### 4.5.1.3 Decoding Strategies [A]
##### Greedy decoding
##### Beam search [F] [A]
##### Length normalization
##### Diverse beam search
##### Sampling (nucleus, temperature)

### 4.5.2 Evaluation Metrics [A]
#### 4.5.2.1 Automatic Metrics [A]
##### BLEU (Bilingual Evaluation Understudy) [A]
###### N-gram precision
###### Brevity penalty
###### Corpus-level vs sentence-level
###### Limitations

##### ROUGE [A]
##### METEOR [A]
##### BERTScore [A]
##### COMET (neural metric)

#### 4.5.2.2 Human Evaluation [A]
##### Adequacy
##### Fluency
##### Ranking

---

## 4.6 Question Answering [A]

### 4.6.1 Extractive QA [A]
#### 4.6.1.1 SQuAD-Style QA [A]
##### Span selection
##### Start & end position prediction
##### BiDAF (Bi-Directional Attention Flow)
##### QANet
##### BERT for QA

#### 4.6.1.2 Inference [A]
##### Paragraph selection
##### Span extraction
##### Confidence scoring

### 4.6.2 Generative QA [A]
#### 4.6.2.1 Models [A]
##### Seq2Seq for QA
##### T5 for QA
##### GPT-style QA
##### Encoder-decoder QA

#### 4.6.2.2 Challenges [A]
##### Hallucination
##### Factual consistency
##### Out-of-context generation

### 4.6.3 Open-Domain QA [A]
#### 4.6.3.1 Two-Stage Approach [A]
##### Document retrieval
##### Reading comprehension
##### Retriever-reader architecture

#### 4.6.3.2 End-to-End Models [A]
##### Dense passage retrieval (DPR)
##### ColBERT (late interaction)
##### Fusion-in-decoder models
##### RAG (Retrieval-Augmented Generation)

---

## 4.7 Sentiment & Opinion Analysis [F]

### 4.7.1 Sentiment Classification [F]
#### 4.7.1.1 Tasks [F]
##### Document-level sentiment
##### Sentence-level sentiment
##### Aspect-based sentiment
##### Target-dependent sentiment
##### Emotion detection

#### 4.7.1.2 Models [F]
##### CNN for text [F]
##### RNN/LSTM for sentiment
##### Attention-based models
##### Transformer-based classifiers
##### Ensemble methods

### 4.7.2 Opinion Extraction [A]
#### 4.7.2.1 Aspect Extraction [A]
##### Dependency parsing
##### CRF-based extraction
##### Joint aspect-sentiment extraction

#### 4.7.2.2 Opinion Mining [A]
##### Opinion expression detection
##### Holder and target identification

---

## 4.8 Summarization [A]

### 4.8.1 Extractive Summarization [A]
#### 4.8.1.1 Methods [A]
##### TF-IDF based selection
##### Graph-based ranking (TextRank, LexRank)
##### Supervised ranking
##### Reinforcement learning for summarization

#### 4.8.1.2 Evaluation [A]
##### ROUGE metrics
##### Human evaluation
##### Redundancy metrics

### 4.8.2 Abstractive Summarization [A]
#### 4.8.2.1 Models [A]
##### Seq2Seq with attention
##### LSTM-based models
##### Transformer-based models (BART, PEGASUS)
##### Copy mechanisms for preserving facts

#### 4.8.2.2 Challenges [A]
##### Factual consistency
##### Hallucination
##### Length control
##### Entity preservation

### 4.8.3 Multi-Document Summarization [A]
#### 4.8.3.1 Approaches [A]
##### Hierarchical models
##### Graph-based fusion
##### Query-focused summarization

---

## 4.9 Relation Extraction & Knowledge Graphs [A]

### 4.9.1 Relation Extraction [A]
#### 4.9.1.1 Binary Relation Extraction [A]
##### Task formulation
##### Sentence-level vs document-level
##### Classification approach
##### End-to-end extraction

#### 4.9.1.2 Methods [A]
##### Supervised classification
##### Semi-supervised learning
##### Distant supervision [A]
##### Reinforcement learning approach

#### 4.9.1.3 Neural Models [A]
##### RNN-based extraction
##### CNN for relations
##### Attention-based models
##### Graph neural networks for relations

### 4.9.2 Knowledge Graphs [A]
#### 4.9.2.1 KG Representation [A]
##### Triple format (head, relation, tail)
##### RDF/Linked Data
##### Graph structure
##### Embedding methods

#### 4.9.2.2 Link Prediction [A]
##### TransE (translation-based)
##### DistMult (multiplicative)
##### ComplEx (complex embeddings)
##### Graph neural networks for link prediction

#### 4.9.2.3 Entity & Relation Linking [A]
##### Entity disambiguation
##### Linking to knowledge bases
##### Candidate generation & ranking

---

## 4.10 Coreference Resolution [A]

### 4.10.1 Coreference Tasks [A]
#### 4.10.1.1 Problem Definition [A]
##### Mention detection
##### Coreference clustering
##### Gold vs predicted mentions

#### 4.10.1.2 Mention Representation [A]
##### Span embeddings
##### Contextual representation
##### Feature engineering

### 4.10.2 Coreference Models [A]
#### 4.10.2.1 Pairwise Models [A]
##### Pair linking
##### Threshold-based clustering
##### Agglomerative clustering

#### 4.10.2.2 Cluster-Based Models [A]
##### Mention ranking
##### Entity linking
##### Joint mention detection & coreference

#### 4.10.2.3 Recent Approaches [A]
##### Span-based LSTM
##### Transformer-based coreference
##### Joint models with other tasks

---

## 4.11 Semantic Role Labeling [A]

### 4.11.1 SRL Formulation [A]
#### 4.11.1.1 Task Definition [A]
##### Predicate identification
##### Argument identification
##### Argument role classification

#### 4.11.1.2 Frameworks [A]
##### PropBank
##### FrameNet
##### VerbNet

### 4.11.2 SRL Models [A]
#### 4.11.2.1 Methods [A]
##### Dependency-based SRL
##### Span-based SRL
##### Sequence tagging approach
##### Neural SRL models

---

## 4.12 Dialogue Systems [A]

### 4.12.1 Dialogue Understanding [A]
#### 4.12.1.1 Intent Detection [A]
##### Classification task
##### Multi-intent scenarios

#### 4.12.1.2 Slot Filling [A]
##### Sequence labeling
##### Named entity recognition variant
##### Joint intent & slot model

#### 4.12.1.3 Dialogue State Tracking [A]
##### DST task formulation
##### Slot values
##### Open-domain tracking
##### Joint multidomain dialogue

### 4.12.2 Dialogue Generation [A]
#### 4.12.2.1 Response Generation [A]
##### Template-based approaches
##### Retrieval-based methods
##### Neural generative models
##### End-to-end dialogue

#### 4.12.2.2 Dialogue Policies [A]
##### Reinforcement learning for dialogue
##### Policy gradient methods
##### Reward modeling
##### Evaluation of dialogue

#### 4.12.2.3 Conversation Context [A]
##### Context encoding
##### Dialogue history
##### Speaker roles
##### Multimodal dialogue

---

# 5. Large Language Models (LLMs)

## 5.1 Transformer Architecture Deep Dive

### 5.1.1 Core Components Review [F] [A]
#### 5.1.1.1 Self-Attention Analysis [F] [A]
##### Attention computation (Q, K, V)
##### Softmax normalization
##### Gradient flow through attention
##### Attention pattern properties
##### Attention head analysis [A]
##### Critical head identification [A]

#### 5.1.1.2 Positional Encoding [F] [A]
##### Sinusoidal encoding properties
##### Rotary Position Embeddings (RoPE) [A]
###### Rotation matrix interpretation
###### Relative position encoding
###### Extrapolation properties

##### Alibi (Attention with Linear Biases) [A]
###### Bias computation
###### Extrapolation beyond training

##### Other position encoding schemes
##### Position interpolation [A]

#### 5.1.1.3 Feed-Forward Networks [A]
##### MLP layers in Transformers
##### Intermediate dimension (4x hidden)
##### Activation function choices
##### Sparse models (MoE) [A]

#### 5.1.1.4 Layer Normalization [A]
##### Post-norm vs pre-norm [A]
##### Gradient flow implications
##### Numerical stability
##### Adaptive computation

---

## 5.2 LLM Pretraining

### 5.2.1 Pretraining Objectives [F] [A]
#### 5.2.1.1 Causal Language Modeling [F]
##### Next-token prediction
##### Teacher forcing
##### Scheduled sampling
##### Loss weighting strategies

#### 5.2.1.2 Masked Language Modeling [A]
##### Random masking
##### Masking strategies
##### Unmasking probabilities

#### 5.2.1.3 Hybrid Objectives [A]
##### Denoising objectives
##### Permutation language modeling
##### Span corruption (T5)
##### Multiple objective balancing

### 5.2.2 Pretraining Data [F] [A]
#### 5.2.2.1 Data Collection & Curation [A]
##### Web data sources (Common Crawl, etc.)
##### Book corpora
##### Code data
##### Academic papers
##### Multilingual data
##### Synthetic data generation [A]

#### 5.2.2.2 Data Preprocessing [A]
##### Quality filtering
##### Deduplication
##### PII removal
##### Bias detection
##### Tokenization scale

#### 5.2.2.3 Data Contamination [A]
##### Benchmark contamination
##### Test set overlap
##### Detection methods
##### Impact assessment

### 5.2.3 Scaling Laws [A]
#### 5.2.3.1 Chinchilla Scaling Laws [A]
##### Model size (parameters)
##### Data size (tokens)
##### Optimal allocation
##### Loss prediction
##### Extrapolation

#### 5.2.3.2 Emergent Abilities [A]
##### In-context learning [A]
##### Few-shot prompting
##### Chain-of-thought reasoning
##### Instruction following
##### Scaling analysis [A]

#### 5.2.3.3 Compute-Optimal Training [A]
##### FLOPs estimation
##### Training efficiency
##### Convergence speed

---

## 5.3 Fine-Tuning Methods [F] [A]

### 5.3.1 Supervised Fine-Tuning (SFT) [F] [A]
#### 5.3.1.1 SFT Setup [F]
##### Instruction-response pairs
##### Dataset creation
##### Prompt templates
##### Few-shot examples in dataset

#### 5.3.1.2 Training Procedure [F]
##### Cross-entropy loss
##### Learning rate scheduling
##### Early stopping
##### Validation metrics
##### Overfitting prevention [A]

#### 5.3.1.3 SFT Challenges [A]
##### Instruction diversity
##### Quality of demonstrations
##### Domain mismatch
##### Catastrophic forgetting

### 5.3.2 Reinforcement Learning from Human Feedback (RLHF) [F] [A]
#### 5.3.2.1 RLHF Framework [A]
##### Stage 1: SFT (initial model)
##### Stage 2: Reward modeling
###### Human preference data
###### Pairwise comparisons
###### Reward model architecture
###### Bradley-Terry model

##### Stage 3: RL training
###### Policy gradient methods
###### PPO (Proximal Policy Optimization) [A]
###### KL penalty
###### Advantage estimation

#### 5.3.2.2 PPO Implementation [A]
##### Mini-batch training
##### Clipped objective
##### Entropy bonus
##### Value function
##### Generalized Advantage Estimation (GAE)

#### 5.3.2.3 RLHF Challenges [A]
##### Reward model generalization [A]
##### Distribution shift
##### Reward hacking [A]
##### Human annotation cost
##### Preference inconsistency

### 5.3.3 Alternative Alignment Methods [A]
#### 5.3.3.1 Direct Preference Optimization (DPO) [A]
##### Implicit reward from preference
##### Bradley-Terry reformulation
##### No separate reward model
##### Training stability
##### Loss function

#### 5.3.3.2 Other Methods [A]
##### Contrastive learning for alignment
##### Self-play fine-tuning
##### Constitutional AI (CAI)
##### Prompt-based alignment

### 5.3.4 Parameter-Efficient Fine-Tuning [A] [S]
#### 5.3.4.1 LoRA (Low-Rank Adaptation) [A]
##### Low-rank decomposition
##### Adapter design
##### Rank selection
##### Scaling matrix
##### Computational efficiency

#### 5.3.4.2 QLoRA [A]
##### Quantization + LoRA
##### 4-bit quantization
##### Memory efficiency
##### Performance trade-offs

#### 5.3.4.3 Other PEFT Methods [A]
##### Adapter layers
##### Prefix tuning
##### Prompt tuning
##### BitFit (bias term tuning)

---

## 5.4 Instruction Tuning & Prompting [F] [A]

### 5.4.1 Instruction Datasets [F] [A]
#### 5.4.1.1 Instruction Format [F]
##### Task instruction
##### Input-output examples
##### Context information
##### Template structure

#### 5.4.1.2 Dataset Construction [A]
##### Crowdsourcing instructions
##### Synthetic instruction generation
##### Task diversity
##### Quality control

#### 5.4.1.3 Public Datasets [A]
##### FLAN
##### Super-NaturalInstructions
##### ALPACA
##### Self-Instruct

### 5.4.2 Prompt Engineering [F] [A]
#### 5.4.2.1 Prompt Design [F]
##### Zero-shot prompting [F]
##### Few-shot prompting [F]
##### In-context learning principles
##### Example selection

#### 5.4.2.2 Advanced Prompting Techniques [A]
##### Chain-of-Thought (CoT) [A]
###### Step-by-step reasoning
###### Intermediate steps
###### CoT generalization
###### Self-consistency [A]

##### Tree-of-Thought [A]
##### Retrieve-Then-Read [A]
##### Scratchpad reasoning [A]
##### Least-to-Most prompting [A]
##### Plan-and-Execute [A]

#### 5.4.2.3 Prompt Optimization [A]
##### Prompt engineering by hand
##### Automatic prompt optimization
##### Prompt distillation [A]
##### Meta-prompt learning

### 5.4.3 Few-Shot Learning in LLMs [A]
#### 5.4.3.1 In-Context Learning [A]
##### Mechanism of in-context learning [A]
##### Example ordering effects
##### Label-only vs full examples
##### Diversity vs specificity

#### 5.4.3.2 Instruction Format Sensitivity [A]
##### Template variations
##### Separators and formatting
##### Instruction clarity
##### Prompt brittleness

---

## 5.5 Decoding Strategies [F] [A]

### 5.5.1 Generation Basics [F]
#### 5.5.1.1 Sampling Strategies [F]
##### Greedy decoding [F]
##### Temperature scaling [F]
##### Top-K sampling [A]
##### Nucleus (Top-P) sampling [A]
##### Tail-free sampling [A]
##### Mirostat sampling [A]

#### 5.5.1.2 Length Control [A]
##### Maximum length
##### Minimum length
##### Length penalties
##### Length normalization

### 5.5.2 Beam Search [F] [A]
#### 5.5.2.1 Beam Search Mechanics [F]
##### Beam width
##### Pruning strategy
##### Length normalization
##### Coverage penalty

#### 5.5.2.2 Beam Search Variants [A]
##### Diverse beam search
##### Constrained beam search
##### Best-first search
##### A* search for generation

### 5.5.3 Deterministic Methods [A]
#### 5.5.3.1 Argmax Decoding [A]
##### Greedy selection
##### Limitations
##### Exposure bias

#### 5.5.3.2 Dynamic Beam Search [A]

### 5.5.4 Generation with Constraints [A]
#### 5.5.4.1 Hard Constraints [A]
##### Format constraints
##### Must-include tokens
##### Forbidden tokens
##### Constrained decoding algorithms

#### 5.5.4.2 Soft Constraints [A]
##### Penalty for violating constraints
##### Weighted preferences
##### Utility functions

---

## 5.6 Context Window & Long-Context Understanding [A]

### 5.6.1 Context Length Limitations [A]
#### 5.6.1.1 Training Context [A]
##### Maximum training sequence length
##### Computational limits (quadratic attention)
##### Memory requirements

#### 5.6.1.2 Inference-Time Extension [A]
##### Position interpolation [A]
##### Rotary embedding extrapolation [A]
##### Context compression techniques

### 5.6.2 Long-Context Methods [A]
#### 5.6.2.1 Architectural Changes [A]
##### Linear attention approximations
##### Sparse attention patterns
##### Sliding window attention
##### Hierarchical attention

#### 5.6.2.2 Training Strategies [A]
##### Continued pretraining on long sequences
##### Retrieval-augmented approaches
##### Efficient fine-tuning

### 5.6.3 Working with Long Documents [A]
#### 5.6.3.1 Retrieval-Based Approach [A]
##### Chunk selection
##### Relevance scoring
##### Combine with long-context models

#### 5.6.3.2 Hierarchical Approaches [A]
##### Summary + details
##### Summarization chains
##### Tree-structured processing

---

## 5.7 Hallucination & Factuality [A]

### 5.7.1 Hallucination Types [A]
#### 5.7.1.1 Classification [A]
##### Intrinsic hallucination (contradicts input)
##### Extrinsic hallucination (unsupported by input)
##### Open-domain hallucination
##### Closed-domain hallucination (with reference)

#### 5.7.1.2 Root Causes [A]
##### Training data issues
##### Decoding strategies
##### Model uncertainty
##### Knowledge gaps

### 5.7.2 Detection Methods [A]
#### 5.7.2.1 Model-Based Detection [A]
##### Likelihood-based approaches
##### Uncertainty estimation
##### Confidence scores
##### Token-level confidence

#### 5.7.2.2 Reference-Based Detection [A]
##### Entailment checking
##### Semantic similarity
##### Information extraction
##### Knowledge base checking

### 5.7.3 Mitigation Strategies [A]
#### 5.7.3.1 Training-Time Approaches [A]
##### High-quality SFT data
##### Fact-grounded pretraining
##### Adversarial fine-tuning
##### Decoding with retrieval

#### 5.7.3.2 Inference-Time Approaches [A]
##### Retrieval-augmented generation (RAG) [A]
##### Fact verification step
##### Self-correction mechanisms
##### Uncertainty-aware decoding

#### 5.7.3.3 Consistency & Grounding [A]
##### Faithfulness to input
##### Long-form factuality
##### Citation generation

---

## 5.8 Model Scaling & Efficiency [A] [S]

### 5.8.1 Parameter Scaling [A]
#### 5.8.1.1 Model Size Variants [A]
##### Small models (1B-7B)
##### Medium models (13B-70B)
##### Large models (100B+)
##### Distributed training implications

#### 5.8.1.2 Scaling Efficiency [A]
##### Memory requirements per parameter
##### Computation-to-memory ratio
##### Bandwidth limitations
##### Latency bottlenecks

### 5.8.2 Inference Optimization [A] [S]
#### 5.8.2.1 Quantization for LLMs [A]
##### INT8 quantization
##### INT4 quantization
##### Group quantization [A]
##### GPTQ method [A]
##### Per-channel vs per-tensor

#### 5.8.2.2 KV-Cache Optimization [A]
##### Key-value caching
##### Memory bottleneck in generation
##### Cache quantization [A]
##### Batch-size considerations

#### 5.8.2.3 Attention Optimization [A]
##### Flash Attention [A]
##### Fused kernels
##### Memory-efficient attention
##### I/O aware algorithms

### 5.8.3 Distributed Inference [A] [S]
#### 5.8.3.1 Strategies [A]
##### Tensor parallelism [A]
##### Pipeline parallelism [A]
##### Sequence parallelism [A]
##### Multi-GPU inference

#### 5.8.3.2 Latency Optimization [A]
##### Batch size vs latency
##### Continuous batching
##### Request scheduling
##### Load balancing

---

## 5.9 Evaluation of LLMs [F] [A]

### 5.9.1 Automatic Evaluation [F]
#### 5.9.1.1 Benchmark-Based [F] [A]
##### MMLU (Massive Multitask Language Understanding)
##### HellaSwag
##### TruthfulQA [A]
##### HumanEval (code)
##### MATH (math reasoning)
##### BIG-bench
##### BLEU, ROUGE, METEOR
##### Custom benchmarks

#### 5.9.1.2 Model-Based Evaluation [A]
##### LLM as judge
##### GPT-4 evaluation
##### Pairwise comparison
##### Bias in model judges
##### Correlation with human evaluation

### 5.9.2 Human Evaluation [F] [A]
#### 5.9.2.1 Evaluation Criteria [F]
##### Correctness / Accuracy
##### Fluency
##### Coherence
##### Relevance
##### Informativeness
##### Harm-related (toxicity, bias)

#### 5.9.2.2 Evaluation Setup [A]
##### Inter-annotator agreement
##### Crowdsourcing vs expert eval
##### Scale of evaluation
##### Cost considerations

#### 5.9.2.3 Specific Evaluations [A]
##### Safety/alignment evaluation
##### Bias & fairness evaluation
##### Robustness testing
##### Adversarial evaluation [A]

### 5.9.3 Behavioral Testing [A]
#### 5.9.3.1 Adversarial Testing [A]
##### Adversarial prompts
##### Jailbreak attempts
##### Edge case generation
##### Robustness to paraphrasing

#### 5.9.3.2 Behavioral Evaluation [A]
##### Task performance consistency
##### Prompt sensitivity
##### Few-shot stability
##### Output diversity

---

## 5.10 Safety, Alignment, & Responsible AI [A]

### 5.10.1 Alignment [A]
#### 5.10.1.1 Alignment Goals [A]
##### Instruction following
##### Harmlessness
##### Honesty / Truthfulness
##### Helpfulness
##### Constitutional AI approach [A]

#### 5.10.1.2 Value Alignment [A]
##### Human values specification
##### Multi-stakeholder perspectives
##### Cultural sensitivity
##### Philosophical alignment

#### 5.10.1.3 Measurement [A]
##### Evaluating alignment
##### Specification gaming detection
##### Reward hacking identification

### 5.10.2 Toxicity & Bias [A]
#### 5.10.2.1 Toxicity Issues [A]
##### Toxicity datasets
##### Toxicity detection
##### Mitigation techniques
##### Red-teaming [A]

#### 5.10.2.2 Bias in LLMs [A]
##### Stereotypes in training data
##### Gender bias
##### Racial bias
##### Occupational bias
##### Bias measurement

#### 5.10.2.3 Bias Mitigation [A]
##### Controlled generation
##### Debiased fine-tuning
##### Inference-time bias mitigation
##### Counterfactual data

### 5.10.3 Privacy & Security [A]
#### 5.10.3.1 Privacy Concerns [A]
##### Memorization of training data
##### Extraction attacks [A]
##### Privacy-preserving training
##### Differential privacy [A]

#### 5.10.3.2 Adversarial Robustness [A]
##### Adversarial examples
##### Prompt injection attacks [A]
##### Robustness evaluation
##### Defense mechanisms

### 5.10.4 Responsible Deployment [A]
#### 5.10.4.1 Transparency [A]
##### Model cards [A]
##### Dataset documentation
##### Limitations disclosure
##### Intended use

#### 5.10.4.2 Monitoring & Auditing [A]
##### Output monitoring
##### User behavior monitoring
##### Bias detection post-deployment
##### Continuous improvement

---

## 5.11 Emerging LLM Research Areas [A]

### 5.11.1 Multimodal LLMs [A]
#### 5.11.1.1 Vision-Language Models [A]
##### CLIP approach (contrastive learning)
##### Image-text pretraining
##### Visual grounding
##### Referring expression

#### 5.11.1.2 Vision-Language-Text Models [A]
##### GPT-4V style models
##### Image understanding + language
##### Instruction-tuned multimodal models
##### Vision instruction tuning

#### 5.11.1.3 Multi-Modality Integration [A]
##### Fusion architectures
##### Cross-modal attention
##### Alignment techniques

### 5.11.2 Tool Use & Agents [A]
#### 5.11.2.1 Function Calling [A]
##### API invocation
##### Tool selection
##### Parameter specification
##### Error handling

#### 5.11.2.2 Reasoning with Tools [A]
##### ReAct (Reasoning + Acting)
##### Planning with tools
##### Multi-step tool use
##### Tool composition

#### 5.11.2.3 Agent Architectures [A]
##### Autonomous agents
##### Goal-oriented agents
##### Multi-agent systems
##### Environment interaction

### 5.11.3 Mixture of Experts (MoE) [A]
#### 5.11.3.1 MoE Concepts [A]
##### Expert networks
##### Gating mechanism
##### Load balancing [A]
##### Router learning

#### 5.11.3.2 MoE in Transformers [A]
##### FFN replacement with MoE
##### Sparse MoE
##### Dense MoE (soft routing)
##### Communication overhead

#### 5.11.3.3 Switch Transformers [A]
##### Simplified MoE
##### Single expert selection
##### Scaling properties
##### Training efficiency

### 5.11.4 Knowledge-Augmented LLMs [A]
#### 5.11.4.1 Knowledge Integration [A]
##### Pre-computed knowledge
##### Knowledge bases
##### Structured knowledge graphs
##### Hybrid retrieval-generation

#### 5.11.4.2 Memory-Augmented Models [A]
##### External memory modules
##### Memory update mechanisms
##### Retrieval strategies

---

# 6. Retrieval-Augmented Generation (RAG)

## 6.1 RAG Fundamentals

### 6.1.1 RAG Framework [F] [A]
#### 6.1.1.1 Core Idea [F]
##### Retrieve relevant documents
##### Augment input with retrieved context
##### Generate answer with context
##### Advantages over pure LLM generation

#### 6.1.1.2 RAG Pipeline [F] [A]
##### Input query
##### Retrieval component
###### Dense retrievers
###### Sparse retrievers
###### Rerankers

##### Context augmentation
##### Generation component
##### Output

#### 6.1.1.3 RAG vs LLM Comparison [A]
##### Factuality improvement [A]
##### Reduced hallucination [A]
##### Knowledge update without retraining [A]
##### Computation trade-offs

---

## 6.2 Retrieval Methods

### 6.2.1 Sparse Retrievers [F]
#### 6.2.1.1 BM25 [F]
##### Term frequency (TF)
##### Inverse document frequency (IDF)
##### Document length normalization
##### Parameter tuning
##### Advantages (efficiency, explainability)
##### Limitations (lexical matching)

#### 6.2.1.2 TF-IDF Variants [F]
##### L2 normalization
##### Sublinear TF scaling
##### Custom IDF weighting
##### Sparse vector representations

#### 6.2.1.3 Advanced Sparse Methods [A]
##### PLaid (Probabilistic Lexicon Augmentation)
##### DeepImpact (learned term weights)
##### uniCOIL (contextual output interaction)

### 6.2.2 Dense Retrievers [F] [A]
#### 6.2.2.1 Dense Passage Retrieval (DPR) [A]
##### Dual encoder architecture
##### Query encoder
##### Passage encoder
##### Contrastive training
##### Negative sampling strategies
##### In-batch negatives

#### 6.2.2.2 Embedding Models [A]
##### Sentence transformers (SBERT)
##### ANCE (Approximate Nearest Neighbor Negative Contrastive Learning)
##### SimCLR-based retrievers
##### Contrastive learning objectives

#### 6.2.2.3 Cross-Encoder Rerankers [A]
##### Ranking rather than retrieval
##### BERT cross-encoder
##### MonoT5 reranker
##### Reranking efficiency trade-off

### 6.2.3 Hybrid Retrieval [A]
#### 6.2.3.1 Combining Methods [A]
##### BM25 + dense vectors
##### Reciprocal Rank Fusion (RRF)
##### Linear combination of scores
##### Learning to combine

#### 6.2.3.2 Late Interaction [A]
##### ColBERT architecture
##### Token-level matching
##### Efficient approximation
##### ColBERTv2 improvements

### 6.2.4 Advanced Retrieval [A]
#### 6.2.4.1 Fusion Methods [A]
##### Fusion-in-Decoder (FiD)
##### Passage fusion
##### Multi-passage encoding
##### Computation requirements

#### 6.2.4.2 Query Expansion [A]
##### Multi-query expansion [A]
##### Generated queries
##### Pseudo-relevance feedback
##### Query rewriting [A]

#### 6.2.4.3 Recursive Retrieval [A]
##### Multi-hop retrieval
##### Iterative refinement
##### Question decomposition

---

## 6.3 Embeddings for RAG

### 6.3.1 Embedding Models [F] [A]
#### 6.3.1.1 Dense Embeddings [F]
##### Dimensionality
##### Normalized vs unnormalized
##### Fixed vs learnable dimensions
##### Model architecture (BiLSTM, Transformers)

#### 6.3.1.2 Popular Embeddings [A]
##### Sentence-BERT [A]
##### OpenAI embeddings
##### Cohere embeddings
##### Open-source models (Jina, BGE, etc.)

#### 6.3.1.3 Embedding Properties [A]
##### Semantic similarity preservation
##### Domain-specific embeddings
##### Multilingual embeddings
##### Long-document embeddings

### 6.3.2 Embedding Fine-Tuning [A]
#### 6.3.2.1 Training Data [A]
##### Positive pairs
##### Hard negatives [A]
##### In-batch negatives
##### Contrastive learning

#### 6.3.2.2 Fine-Tuning Methods [A]
##### Triplet loss
##### InfoNCE loss
##### In-batch softmax
##### Curriculum learning

#### 6.3.2.3 Domain Adaptation [A]
##### Task-specific fine-tuning
##### Few-shot adaptation
##### Transfer learning for embeddings

### 6.3.3 Embedding Quantization [A]
#### 6.3.3.1 Quantization Methods [A]
##### Bit quantization
##### Product quantization [A]
##### Learned quantization [A]

#### 6.3.3.2 Trade-offs [A]
##### Accuracy vs compression
##### Speed improvements
##### Memory savings

---

## 6.4 Vector Databases & Indexing

### 6.4.1 Vector Database Fundamentals [A]
#### 6.4.1.1 Core Features [A]
##### Vector similarity search
##### Semantic search capability
##### Metadata filtering
##### Update operations
##### Scalability

#### 6.4.1.2 Popular Vector DBs [A]
##### Pinecone
##### Weaviate
##### Milvus
##### Qdrant
##### ChromaDB
##### Faiss (Facebook AI Similarity Search)

#### 6.4.1.3 Integration with LLMs [A]
##### LLM-vector DB coupling
##### Index refresh strategies
##### Version control for indices

### 6.4.2 Indexing Methods [A]
#### 6.4.2.1 Exact Search [A]
##### Brute force search
##### Kd-tree
##### Ball tree
##### When to use exact

#### 6.4.2.2 Approximate Search [A]
##### HNSW (Hierarchical Navigable Small World)
##### LSH (Locality-Sensitive Hashing)
##### PQ (Product Quantization)
##### IVF (Inverted File)

#### 6.4.2.3 Efficiency Considerations [A]
##### Index construction time
##### Query latency
##### Memory footprint
##### Update efficiency

### 6.4.3 Retrieval at Scale [A] [S]
#### 6.4.3.1 Distributed Retrieval [A]
##### Sharding strategies
##### Replica management
##### Consistency guarantees

#### 6.4.3.2 Performance Optimization [A]
##### Batch retrieval
##### Caching strategies
##### Prefetching
##### Request scheduling

---

## 6.5 Chunking Strategies [A]

### 6.5.1 Document Splitting [A]
#### 6.5.1.1 Chunk Size Selection [A]
##### Fixed-size chunks
##### Variable-size chunks
##### Semantic chunks [A]
##### Trade-offs (coverage vs coherence)

#### 6.5.1.2 Overlap Strategies [A]
##### No overlap
##### Fixed overlap
##### Content-aware overlap
##### Boundary detection

#### 6.5.1.3 Chunking Methods [A]
##### Character-based splitting
##### Token-based splitting
##### Sentence-based splitting
##### Paragraph-based splitting
##### Recursive splitting

### 6.5.2 Semantic Chunking [A]
#### 6.5.2.1 Topic-Based [A]
##### Coherence scoring
##### Topic changes detection
##### Semantic boundaries

#### 6.5.2.2 Discourse-Aware [A]
##### Structure utilization (sections, paragraphs)
##### Heading-based chunking
##### Sentence embedding similarity

### 6.5.3 Chunk Quality & Metadata [A]
#### 6.5.3.1 Rich Metadata [A]
##### Document source
##### Chunk position
##### Section/category
##### Temporal information

#### 6.5.3.2 Chunk Filtering [A]
##### Remove low-quality chunks
##### Duplicate detection
##### Content-based filtering

---

## 6.6 Query Processing & Rewriting [A]

### 6.6.1 Query Enhancement [A]
#### 6.6.1.1 Query Expansion [A]
##### Synonym expansion
##### Question paraphrasing
##### Query decomposition [A]
##### Sub-question generation

#### 6.6.1.2 Query Rewriting [A]
##### Clarification requests
##### Context-aware rewriting
##### Temporal reasoning
##### Implicit information extraction

### 6.6.2 Multi-Query Generation [A]
#### 6.6.2.1 Approach [A]
##### Generate multiple query variants
##### Ensemble retrieval
##### Diversity in queries [A]

#### 6.6.2.2 Implementation [A]
##### LLM-based generation
##### Template-based variants
##### Retrieval with all variants

---

## 6.7 Ranking & Reranking [A]

### 6.7.1 Reranking Strategies [A]
#### 6.7.1.1 Cross-Encoder Reranking [A]
##### Passage relevance prediction
##### Ranking score refinement
##### Computational overhead
##### Latency considerations

#### 6.7.1.2 Diversity-Aware Reranking [A]
##### Maximal Marginal Relevance (MMR)
##### Diversity-relevance trade-off
##### Sub-modular optimization

#### 6.7.1.3 Fusion & Ensemble [A]
##### Combine multiple rankers
##### Score normalization
##### Weighted combination

### 6.7.2 Learning-to-Rank [A]
#### 6.7.2.1 Approaches [A]
##### Pointwise ranking
##### Pairwise ranking
##### Listwise ranking
##### LambdaMART

#### 6.7.2.2 Application to RAG [A]
##### Training data generation
##### Online hard negative mining
##### Relevance label types

---

## 6.8 RAG Evaluation [A]

### 6.8.1 Retrieval Evaluation [A]
#### 6.8.1.1 Metrics [A]
##### Recall@K [A]
##### MRR (Mean Reciprocal Rank) [A]
##### NDCG (Normalized Discounted Cumulative Gain) [A]
##### Precision@K
##### MAP (Mean Average Precision)

#### 6.8.1.2 Evaluation Challenges [A]
##### Sparse judgments
##### Relevance ambiguity
##### Scale of evaluation

### 6.8.2 End-to-End RAG Evaluation [A]
#### 6.8.2.1 Generation Quality [A]
##### Answer correctness
##### Faithfulness to retrieved documents
##### Hallucination detection
##### Citation accuracy

#### 6.8.2.2 Metrics [A]
##### Combination of retrieval + generation metrics
##### Token-overlap metrics (BLEU, ROUGE)
##### Semantic similarity (BERTScore)
##### Factual correctness (QA evaluation)

#### 6.8.2.3 Benchmark Datasets [A]
##### Natural Questions
##### MS MARCO
##### DPR datasets
##### Custom domain benchmarks

### 6.8.3 Failure Analysis [A]
#### 6.8.3.1 Error Types [A]
##### Retrieval failures
##### Ranking failures
##### Generation errors despite good retrieval
##### Conflicting information handling

#### 6.8.3.2 Debugging [A]
##### Step-by-step analysis
##### Component isolation
##### Error categorization
##### Continuous improvement

---

## 6.9 RAG Failure Modes & Solutions [A]

### 6.9.1 Common Failure Patterns [A]
#### 6.9.1.1 Retrieval Failures [A]
##### No relevant documents in database
##### Query-document mismatch
##### Ambiguous queries
##### Misspellings or language variations

#### 6.9.1.2 Ranking Failures [A]
##### Relevant documents ranked low
##### Irrelevant documents ranked high
##### Conflicting documents
##### Long-tail queries

#### 6.9.1.3 Generation Failures [A]
##### Hallucination despite good retrieval
##### Misinterpretation of context
##### Information synthesis errors
##### Format violations

### 6.9.2 Solutions & Mitigations [A]
#### 6.9.2.1 Retrieval Improvement [A]
##### Better chunking
##### Enhanced embeddings
##### Query expansion
##### Multi-step retrieval

#### 6.9.2.2 Generation Improvement [A]
##### Few-shot prompting in retrieval-augmented context
##### Explicit faithfulness constraints
##### Citation generation
##### Verification steps

#### 6.9.2.3 System-Level Solutions [A]
##### Feedback loops
##### Human-in-the-loop
##### Continuous reranking
##### Iterative refinement

---

## 6.10 Advanced RAG Techniques [A]

### 6.10.1 Retrieval Augmentation Variants [A]
#### 6.10.1.1 Iterative Retrieval [A]
##### Multi-hop reasoning
##### Dependency-aware retrieval
##### Progressive refinement
##### Stop criteria

#### 6.10.1.2 Graph-Based Retrieval [A]
##### Knowledge graph traversal
##### Entity linking
##### Relation extraction
##### Graph reasoning

#### 6.10.1.3 Hybrid Approaches [A]
##### Combine LLM + retrieval + reasoning
##### Modular systems
##### Orchestration patterns

### 6.10.2 Self-Aware RAG [A]
#### 6.10.2.1 Uncertainty Estimation [A]
##### Know when to retrieve
##### Confidence in generation
##### Confidence in retrieved documents

#### 6.10.2.2 Active Retrieval [A]
##### Query selection for retrieval
##### Routing to retrieval components
##### Skip retrieval when confident

### 6.10.3 Training-Aware RAG [A]
#### 6.10.3.1 Joint Training [A]
##### End-to-end optimization
##### Gradient flow to retriever
##### Gradient flow to ranker

#### 6.10.3.2 Adapter-Based RAG [A]
##### Lightweight integration
##### Domain-specific adaptation
##### Multiple specialized retrievers

---

# 7. ML Systems Design & Deployment

## 7.1 End-to-End ML Pipeline

### 7.1.1 Pipeline Architecture [F] [S]
#### 7.1.1.1 Components [F] [S]
##### Data ingestion
##### Data processing & transformation
##### Feature engineering
##### Model training
##### Model evaluation
##### Model deployment
##### Monitoring & feedback

#### 7.1.1.2 Orchestration [S]
##### Workflow scheduling
##### DAGs (Directed Acyclic Graphs)
##### Dependencies between stages
##### Error handling
##### Retry logic

### 7.1.2 Data Infrastructure [F] [S]
#### 7.1.2.1 Data Collection [F] [S]
##### Logging infrastructure
##### Event tracking
##### Sensor data
##### API data sources
##### Third-party data
##### Data governance

#### 7.1.2.2 Data Storage [S]
##### Data lakes (raw data)
##### Data warehouses (structured)
##### Batch vs streaming
##### Partitioning strategies
##### Data lifecycle management

#### 7.1.2.3 Data Quality [F] [S]
##### Data validation
##### Schema enforcement
##### Null value handling
##### Outlier detection
##### Data profiling
##### Great Expectations framework

---

## 7.2 Feature Engineering at Scale [F] [S]

### 7.2.1 Feature Stores [A] [S]
#### 7.2.1.1 Feature Store Concept [A]
##### Centralized feature management
##### Training vs serving features
##### Feature lineage tracking
##### Version control for features
##### Feature discovery

#### 7.2.1.2 Popular Platforms [A]
##### Feast
##### Tecton
##### Databricks Feature Store
##### Hopsworks
##### Custom implementations

#### 7.2.1.3 Feature Store Architecture [A]
##### Offline store (batch)
##### Online store (real-time)
##### Latency requirements
##### Consistency guarantees

### 7.2.2 Feature Engineering Pipeline [F] [S]
#### 7.2.2.1 Batch Feature Engineering [S]
##### Historical feature computation
##### Aggregations
##### Time-window features
##### External data joins

#### 7.2.2.2 Real-Time Features [A] [S]
##### Streaming feature computation
##### Point-in-time correctness
##### State management
##### Low-latency requirements

#### 7.2.2.3 Feature Engineering Tools [S]
##### PySpark for distributed processing
##### Pandas for smaller data
##### Polars for speed
##### DuckDB for analytics

### 7.2.3 Feature Management [F] [S]
#### 7.2.3.1 Feature Versioning [A] [S]
##### Track feature logic changes
##### Backward compatibility
##### Gradual rollout

#### 7.2.3.2 Feature Monitoring [A] [S]
##### Distribution shifts in features
##### Missing values
##### Invalid ranges
##### Correlation changes

---

## 7.3 Model Training Infrastructure [F] [S]

### 7.3.1 Training Frameworks [F] [S]
#### 7.3.1.1 Popular Frameworks [F]
##### TensorFlow [F]
##### PyTorch [F]
##### JAX
##### Keras (high-level API)
##### Scikit-learn (traditional ML)
##### XGBoost, LightGBM (tree-based)

#### 7.3.1.2 Framework Comparison [S]
##### Ease of use
##### Performance
##### Production readiness
##### Ecosystem
##### Community support

### 7.3.2 Distributed Training [A] [S]
#### 7.3.2.1 Distributed Strategies [A]
##### Data parallelism [A]
###### Synchronous updates
###### Asynchronous updates
###### Gradient aggregation
###### Ring all-reduce [A]

##### Model parallelism [A]
###### Pipeline parallelism [A]
###### Tensor parallelism [A]
###### Sequence parallelism [A]

##### Hybrid parallelism [A]

#### 7.3.2.2 Distributed Framework Considerations [A]
##### Communication overhead
##### Synchronization barriers
##### Fault tolerance
##### Checkpointing strategy

#### 7.3.2.3 Tools & Platforms [S]
##### Horovod
##### Ray Tune
##### PyTorch Distributed
##### TensorFlow Distributed
##### DeepSpeed [A]
##### Megatron-LM [A]

### 7.3.3 Hyperparameter Management [F] [S]
#### 7.3.3.1 Hyperparameter Optimization [F]
##### Grid search
##### Random search
##### Bayesian optimization [A]
##### Hyperband (adaptive resource allocation) [A]
##### Population-based training (PBT) [A]

#### 7.3.3.2 Tools [S]
##### Optuna
##### Ray Tune
##### Hyperopt
##### Wandb for tracking
##### MLflow for experiment management

### 7.3.4 Experiment Tracking [F] [S]
#### 7.3.4.1 Tracking Components [S]
##### Hyperparameters
##### Metrics
##### Models & artifacts
##### Code versions
##### Data versions
##### Environment

#### 7.3.4.2 Experiment Management Platforms [S]
##### Weights & Biases (Wandb)
##### MLflow
##### Neptune
##### Kubeflow
##### Guild AI

---

## 7.4 Model Versioning & Registry [A] [S]

### 7.4.1 Model Versioning [A] [S]
#### 7.4.1.1 Version Control [A]
##### Semantic versioning
##### Release notes
##### Model card documentation
##### Backward compatibility

#### 7.4.1.2 Model Artifacts [A]
##### Serialization formats (pickle, ONNX, SavedModel)
##### Dependency freezing
##### Configuration files
##### Metadata storage

### 7.4.2 Model Registry [A] [S]
#### 7.4.2.1 Model Store Concept [A]
##### Central repository
##### Version tracking
##### Metadata
##### Lineage tracking

#### 7.4.2.2 Popular Tools [S]
##### MLflow Model Registry
##### Hugging Face Model Hub
##### DVC (Data Version Control)
##### Custom solutions

#### 7.4.2.3 Registry Operations [A]
##### Register models
##### Promote to staging/production
##### Archive deprecated models
##### Search & discovery

---

## 7.5 Continuous Integration/Continuous Deployment (CI/CD) for ML [A] [S]

### 7.5.1 ML-Specific CI/CD [A] [S]
#### 7.5.1.1 Continuous Integration [A]
##### Code testing
##### Data quality checks
##### Model validation
##### Integration tests
##### Artifact generation

#### 7.5.1.2 Continuous Deployment [A]
##### Model serving
##### Canary deployments
##### A/B testing integration
##### Rollback strategies

#### 7.5.1.3 CI/CD Pipelines [S]
##### Trigger conditions
##### Automated testing stages
##### Approval gates
##### Manual intervention points

### 7.5.2 ML-Specific Testing [A] [S]
#### 7.5.2.1 Model Testing [A]
##### Unit tests for preprocessing
##### Data validation tests
##### Model output validation
##### Performance regression tests
##### Fairness tests

#### 7.5.2.2 Data Testing [A]
##### Schema validation
##### Statistical properties
##### Freshness checks
##### Integrity constraints

#### 7.5.2.3 Integration Testing [A]
##### End-to-end tests
##### Inference correctness
##### Latency requirements
##### Resource constraints

### 7.5.3 Tools & Platforms [S]
##### GitLab CI/CD
##### GitHub Actions
##### Jenkins
##### CircleCI
##### Kubeflow Pipelines
##### Apache Airflow

---

## 7.6 Model Serving & Inference [F] [S]

### 7.6.1 Serving Strategies [F] [S]
#### 7.6.1.1 Batch Inference [F] [S]
##### Offline scoring
##### Scheduled jobs
##### Latency tolerance
##### Throughput focus
##### Examples: daily recommendations, batch predictions

#### 7.6.1.2 Real-Time Inference [F] [S]
##### Low-latency requirements
##### Request-response patterns
##### Scale considerations
##### Examples: recommendation at click time, fraud detection

#### 7.6.1.3 Streaming Inference [A] [S]
##### Continuous processing
##### State management
##### Stream processors (Kafka, Flink)
##### Windowed operations

### 7.6.2 Inference Frameworks [F] [S]
#### 7.6.2.1 Model Serving Platforms [S]
##### TensorFlow Serving [S]
##### TorchServe [S]
##### KServe [S]
##### Seldon [S]
##### BentoML
##### MLflow Models
##### Ray Serve

#### 7.6.2.2 Framework Features [S]
##### Model loading & initialization
##### Request batching
##### Model versioning
##### A/B testing support
##### Canary deployments
##### Monitoring integration

### 7.6.3 Containerization & Orchestration [S]
#### 7.6.3.1 Containerization [S]
##### Docker containers
##### Image optimization
##### Dependency management
##### Multi-stage builds

#### 7.6.3.2 Orchestration [S]
##### Kubernetes [S]
##### Horizontal Pod Autoscaling (HPA)
##### Resource requests/limits
##### Service discovery

#### 7.6.3.3 Deployment Patterns [S]
##### Blue-green deployment
##### Canary deployment [A]
##### Rolling deployment
##### Shadow mode [A]

### 7.6.4 Optimization for Serving [S]
#### 7.6.4.1 Batching [S]
##### Request batching
##### Dynamic batching
##### Batch size selection
##### Latency SLA constraints

#### 7.6.4.2 Caching [S]
##### Output caching
##### Feature caching
##### Cache invalidation
##### TTL strategies

#### 7.6.4.3 Quantization & Compression [A] [S]
##### Model size reduction
##### Inference acceleration
##### Memory efficiency
##### Accuracy trade-offs [A]

---

## 7.7 Monitoring & Observability [F] [S]

### 7.7.1 Model Monitoring [F] [S]
#### 7.7.1.1 Performance Metrics [F]
##### Prediction accuracy
##### Latency
##### Throughput
##### Error rates
##### SLA compliance

#### 7.7.1.2 Data Monitoring [A] [S]
##### Input data distribution
##### Feature statistics
##### Missing values
##### Statistical anomalies
##### Data quality metrics

#### 7.7.1.3 Monitoring Tools [S]
##### Prometheus (metrics collection)
##### Grafana (visualization)
##### ELK Stack (logging)
##### DataDog
##### New Relic
##### Custom dashboards

### 7.7.2 Drift Detection [A] [S]
#### 7.7.2.1 Distribution Shift Detection [A]
##### Covariate shift detection
##### Label shift detection
##### Concept drift detection
##### Statistical tests [A]

#### 7.7.2.2 Drift Monitoring Strategy [A]
##### Baseline period
##### Comparison metrics
##### Alerting thresholds
##### Response automation

#### 7.7.2.3 Tools & Libraries [A]
##### Evidently AI
##### WhyLabs
##### deepchecks
##### Custom implementations

### 7.7.3 Logging & Debugging [S]
#### 7.7.3.1 Comprehensive Logging [S]
##### Input features
##### Model predictions
##### Inference time
##### Errors and exceptions
##### User actions (for feedback)

#### 7.7.3.2 Log Analysis [S]
##### Error rate tracking
##### Performance trends
##### Data quality issues
##### Debugging difficult cases

---

## 7.8 Model Retraining & Updates [A] [S]

### 7.8.1 Retraining Strategies [A]
#### 7.8.1.1 Schedule-Based Retraining [A]
##### Fixed schedule (daily, weekly, monthly)
##### Pros: predictable, simple
##### Cons: may be unnecessary or insufficient

#### 7.8.1.2 Performance-Based Retraining [A]
##### Trigger on performance degradation
##### Trigger on drift detection
##### Threshold-based triggers
##### Continuous vs episodic

#### 7.8.1.3 Data-Based Retraining [A]
##### Retraining when sufficient new data
##### Labeled data collection
##### Active learning for efficient labeling
##### Feedback loops [A]

### 7.8.2 Incremental Learning [A]
#### 7.8.2.1 Online Learning [A]
##### Model updates from single or few examples
##### Concept drift handling
##### Catastrophic forgetting prevention
##### Stability-plasticity trade-off

#### 7.8.2.2 Mini-Batch Retraining [A]
##### Periodic updates with new data
##### Warm-start from previous model
##### Efficient computation

### 7.8.3 A/B Testing for Model Updates [F] [A]
#### 7.8.3.1 Comparing Models [A]
##### Champion vs challenger
##### Metrics for comparison
##### Statistical significance
##### Business impact

#### 7.8.3.2 Rollout Strategy [A]
##### Gradual rollout
##### Shadow mode deployment
##### Region-based rollout
##### User cohort-based rollout

---

## 7.9 Scalability & Distributed Systems [A] [S]

### 7.9.1 Scaling Training [A] [S]
#### 7.9.1.1 Horizontal Scaling [A]
##### Multi-machine training
##### Data parallelism
##### Communication costs
##### Scaling efficiency

#### 7.9.1.2 Vertical Scaling [A]
##### Larger machines
##### GPU/TPU utilization
##### Memory constraints
##### Cost vs performance

#### 7.9.1.3 Training Efficiency [A]
##### Gradient accumulation
##### Mixed precision training (FP16)
##### Gradient checkpointing
##### Layer freezing

### 7.9.2 Scaling Inference [A] [S]
#### 7.9.2.1 Request Processing [S]
##### Concurrent request handling
##### Load balancing
##### Queue management
##### Request routing

#### 7.9.2.2 Model Replication [S]
##### Multiple model replicas
##### Stateless design
##### Session management
##### Consistency guarantees

#### 7.9.2.3 Caching & Acceleration [A] [S]
##### Redis for caching
##### CDN for content delivery
##### Hardware acceleration (GPU, TPU, FPGA)

### 7.9.3 Cost Optimization [A] [S]
#### 7.9.3.1 Computational Cost [A]
##### Cloud pricing models
##### Reserved instances vs spot instances
##### Right-sizing
##### Resource utilization monitoring

#### 7.9.3.2 Storage Cost [A]
##### Data compression
##### Archival strategies
##### Deduplication
##### Tiered storage

#### 7.9.3.3 Optimization Techniques [A]
##### Model compression
##### Inference acceleration
##### Batch processing
##### Scheduling optimization

---

## 7.10 Production Best Practices [F] [S]

### 7.10.1 Code Organization [S]
#### 7.10.1.1 Project Structure [S]
##### Modular design
##### Separation of concerns
##### Reusable components
##### Configuration management

#### 7.10.1.2 Code Quality [S]
##### Testing (unit, integration, e2e)
##### Code reviews
##### Linting & formatting
##### Documentation

### 7.10.2 Reproducibility [F] [S]
#### 7.10.2.1 Reproducible Training [S]
##### Random seed management
##### Deterministic operations
##### Environment pinning (Python version, libraries)
##### GPU non-determinism handling [A]

#### 7.10.2.2 Artifact Management [S]
##### Code version (git)
##### Data version (DVC, git-lfs)
##### Model version
##### Environment specification (requirements.txt, conda.yml)
##### Configuration version

### 7.10.3 Documentation [F] [S]
#### 7.10.3.1 Model Documentation [A] [S]
##### Model card [A] [S]
##### Training data documentation
##### Model limitations
##### Intended use cases
##### Ethical considerations

#### 7.10.3.2 Code Documentation [S]
##### Docstrings
##### Type hints
##### Comments for complex logic
##### README files

### 7.10.4 Security & Privacy [A] [S]
#### 7.10.4.1 Data Security [A]
##### Data encryption at rest
##### Data encryption in transit
##### Access control
##### Audit logging
##### PII handling

#### 7.10.4.2 Model Security [A]
##### Adversarial robustness [A]
##### Model extraction prevention
##### Backdoor detection
##### Poisoning resilience [A]

#### 7.10.4.3 Privacy-Preserving ML [A]
##### Differential privacy
##### Federated learning
##### Secure multi-party computation
##### Homomorphic encryption

---

# 🚨 CRITICAL GAP IDENTIFICATION SECTION

## Commonly Overlooked but High-Impact Topics

### A. Mathematical & Theoretical Gaps

#### A.1 Generalization Theory [A] [T]
- PAC learning bounds
- VC dimension
- Rademacher complexity
- Algorithmic stability
- Cross-validation theory
- Structural risk minimization

#### A.2 Optimization Landscape [A] [T]
- Mode connectivity
- Loss landscape geometry
- Implicit bias of SGD
- Lottery ticket hypothesis mechanics
- Neural tangent kernel theory
- Mean field theory of neural networks

#### A.3 Information Geometry [A] [T]
- Fisher information matrix
- Natural gradient descent
- Manifold optimization
- Geodesic distance in parameter space
- KL divergence geometry

---

### B. Data & Fairness Deep Topics

#### B.1 Data Valuation [A]
- Shapley value for data
- Leave-one-out error
- Data replication values
- Core-set selection
- Active learning theory
- Coreset construction

#### B.2 Fairness Technical Implementation [A]
- Fairness constraints in optimization
- Lagrangian multipliers for fairness
- Pareto frontier in fairness-accuracy
- Intersectional fairness
- Fairness in ranking systems
- Temporal fairness

#### B.3 Label Quality Issues [A]
- Weak supervision
- Noisy labels handling
- Crowdsourcing aggregation
- Annotator bias
- Label noise correction
- Confident learning

---

### C. Specialized Model Architectures Often Missed

#### C.1 Graph Neural Networks [A]
- Graph convolution operations
- Message passing framework
- Graph attention networks
- Spectral graph theory
- Node embedding
- Graph pooling
- Temporal graph networks
- Knowledge graph embeddings

#### C.2 Capsule Networks [A]
- Capsule concept
- Routing algorithms
- Agreement mechanism
- Routing by agreement
- Dynamic routing

#### C.3 Neural ODE & Continuous Models [A]
- Differential equations as models
- ODE solvers in neural networks
- Neural flow models
- Adjoint method
- Continuous normalizing flows

---

### D. Advanced Optimization & Numerical Methods

#### D.1 Second-Order Optimization [A]
- Natural gradient methods
- K-FAC approximation
- Block diagonal Hessian
- Shampoo algorithm
- Preconditioned gradient descent

#### D.2 Stochastic Approximation [T]
- Convergence analysis
- Polyak averaging
- Variance reduction techniques (SVRG, SAGA)
- Proximal algorithms
- Mirror descent

#### D.3 Byzantine-Robust Optimization [A]
- Byzantine gradient descent
- Robust aggregation
- Communication-efficient learning
- Fault tolerance

---

### E. Modern Techniques in LLMs/GenAI Frequently Under-Covered

#### E.1 Speculative Decoding [A]
- Faster generation
- Draft model
- Verification
- Acceptance criterion
- Compute-latency trade-off

#### E.2 Mixture of Experts (MoE) Advanced [A]
- Load balancing techniques
- Expert specialization
- Auxiliary losses
- Router training
- Expert dropout

#### E.3 Parameter Sharing & Adapter Techniques [A]
- Shared vs task-specific parameters
- Bottleneck adapters
- Hypernetworks
- Module adaptation
- Efficient scaling

#### E.4 Training Dynamics in LLMs [A]
- Token predicting learning curves
- Phase transitions in training
- Grokking phenomenon [A]
- Training instability
- KL annealing schedules

---

### F. Systems & Infrastructure Often Missed

#### F.1 Inference Serving Patterns [A] [S]
- Request-level batching
- Feature pre-computation
- Ensemble strategies
- Fallback mechanisms
- Traffic shaping

#### F.2 Data Validation in Production [A]
- Schema evolution
- Backward compatibility
- Data type validation
- Statistical validation
- Constraint checking

#### F.3 Experiment Infrastructure [A]
- Traffic splitting for experiments
- Variance reduction techniques (CUPED)
- Cross-experiment interference
- Long-term effects
- Multi-armed bandit interactions

---

### G. Critical Niche But Frequently Asked

#### G.1 Cold Start Problem [A]
- In recommendation systems
- In RL settings
- Content-based approaches
- Hybrid approaches
- User feedback collection

#### G.2 Click-Through Rate (CTR) Prediction [A]
- Feature engineering for CTR
- Sparse feature handling
- Temporal dynamics
- Cross-feature interactions
- Deep learning CTR models

#### G.3 Learning-to-Rank [A]
- Pointwise/pairwise/listwise approaches
- NDCG optimization
- LambdaRank, LambdaMART
- Position bias handling
- Online learning for ranking

#### G.4 Time Series Specific [A]
- Seasonality modeling
- Trend decomposition
- Autocorrelation
- ARIMA models
- Prophet models
- LSTM for forecasting
- Transformer architectures for time series
- Anomaly detection in time series

---

### H. Interdisciplinary Knowledge Gaps

#### H.1 Causal ML Applied [A]
- Treatment effect heterogeneity
- Causal forests
- Synthetic controls
- Difference-in-differences
- Instrumental variables

#### H.2 Bayesian ML in Practice [A]
- Posterior sampling
- Uncertainty quantification
- Thompson sampling
- Bayesian optimization
- Gaussian processes
- Bayesian neural networks implementation

#### H.3 Reinforcement Learning Basics [A]
- Markov Decision Processes
- Policy gradient methods
- Value iteration
- Q-learning
- Multi-armed bandits
- Contextual bandits

#### H.4 Econometrics for ML [A]
- Causal inference
- Regression discontinuity
- Propensity score matching
- Instrumental variables
- Difference-in-differences

---

### I. Production Reliability Blind Spots

#### I.1 Model Governance [A] [S]
- Model approval workflows
- Risk assessment
- Bias auditing
- Change management
- Model sunset policies

#### I.2 Operational Excellence [A] [S]
- SLA management
- Incident response
- Postmortem processes
- Monitoring coverage
- Alert fatigue prevention

#### I.3 Compliance & Regulatory [A]
- GDPR implications
- Right to explanation
- Model fairness audits
- Documentation requirements
- Audit trails

---

### J. Emerging Research Topics (Cutting Edge)

#### J.1 Efficient LLMs [A]
- Quantization techniques
- Pruning strategies
- Distillation for LLMs
- Sparse models
- Low-rank adaptation

#### J.2 Multimodal Learning [A]
- Vision-language alignment
- Audio-text-image fusion
- Cross-modal retrieval
- Multimodal embeddings
- Video understanding

#### J.3 Prompt Learning [A]
- In-context learning mechanisms
- Prompt optimization
- Gradient-based prompt learning
- Prompt ensembles
- Prompt transferability

#### J.4 Interpretable ML [A]
- Concept-based explanations
- Influence functions
- Attention interpretation [A]
- Probing neural networks
- Mechanistic interpretability

---

### K. Domain-Specific Specializations Often Missed

#### K.1 Recommendation Systems [A]
- Collaborative filtering
- Content-based filtering
- Hybrid approaches
- Diversity in recommendations
- Temporal dynamics
- Cold start solutions

#### K.2 Natural Language Understanding [A]
- Semantic parsing
- Knowledge extraction
- Temporal reasoning
- Spatial reasoning
- Common sense reasoning

#### K.3 Computer Vision Specializations [A]
- Object detection
- Instance segmentation
- Panoptic segmentation
- 3D vision
- Video understanding
- Person re-identification

#### K.4 Audio & Speech [A]
- Acoustic modeling
- Speech recognition
- Speech synthesis
- Audio embeddings
- Music information retrieval

---

### L. Critical Interview Topics Frequently Under-Emphasized

#### L.1 End-to-End System Design Patterns [A] [S]
- Real-world constraint handling
- Latency-accuracy trade-offs
- Cost optimization
- Scalability planning
- Graceful degradation

#### L.2 Statistical Testing for ML [A]
- Multiple comparisons corrections
- Statistical power
- Effect size calculations
- Sequential testing
- Bayesian vs frequentist approaches

#### L.3 Active Learning [A]
- Query selection strategies
- Uncertainty sampling
- Query-by-committee
- Expected model change
- Pool-based vs stream-based

#### L.4 Transfer Learning [A]
- Domain adaptation
- Fine-tuning strategies
- Feature extraction
- Task similarity
- Negative transfer prevention

---

## Strategic Gaps by Interview Type

### For FAANG Interviews [A]
- Heavy emphasis on system design
- Scaling challenges
- Production readiness
- Cost optimization
- Monitoring & debugging

### For ML Engineer Roles [A]
- Infrastructure & DevOps
- Model deployment
- Serving strategies
- Monitoring & alerting
- Data pipelines

### For Data Scientist Roles [A]
- Statistical testing
- Experimentation design
- Causal inference
- Feature engineering edge cases
- Business metrics

### For GenAI/LLM Roles [A]
- Transformer internals
- Fine-tuning methods (RLHF, DPO)
- RAG architectures
- Prompt engineering
- Hallucination mitigation

### For Research Scientist [A]
- Theoretical foundations
- Novel algorithms
- Cutting-edge architectures
- Empirical analysis
- Reproducibility standards

---

## Actionable Gap-Closing Strategies

### Daily Review Topics
1. One complex algorithm (e.g., CoT decoding, Spectral Normalization)
2. One system design pattern (e.g., feature store architecture)
3. One failure mode (e.g., data leakage scenario)
4. One paper summary (latest arXiv)

### Weekly Deep Dives
1. Implement a non-trivial concept
2. Read a seminal paper end-to-end
3. Solve a system design interview
4. Code a complete mini-project

### Monthly Reviews
1. Comprehensive knowledge gap assessment
2. Update mental model with new techniques
3. Practice explaining concepts to others
4. Track emerging trends

---

**END OF COMPREHENSIVE ML KNOWLEDGE TREEMAP**

*Document Version 1.0*
*Last Updated: March 2026*
*Scope: 1000+ nodes across 7 major domains*
*Target: FAANG-level interview preparation*

# ML + Data Science + GenAI Interview Tree Map
## Quick Revision Cheat Sheet for Technical Interviews

---

# Machine Learning

## Supervised Learning ⭐
- **Regression**
  - Linear Regression ⭐
    - OLS, Normal Equation
    - Assumptions (LINE) ⚠️
    - Multicollinearity 🔥
  - Regularized Regression ⭐🧠
    - Ridge (L2) — handles multicollinearity
    - Lasso (L1) — feature selection
    - Elastic Net — combines both
  - Logistic Regression ⭐
    - Sigmoid, Log-Odds
    - Threshold tuning
  - Polynomial Regression
    - Overfitting risk ⚠️

- **Classification**
  - Tree-Based ⭐
    - Decision Trees
      - Gini vs Entropy
      - Pruning strategies
    - Random Forest ⭐
      - Bagging, Feature randomness
      - OOB score
    - Gradient Boosting ⭐🔥
      - XGBoost / LightGBM / CatBoost
      - Learning rate, Early stopping
  - SVM ⭐🧠
    - Kernel trick
    - Soft vs Hard margin
    - C parameter trade-off
  - KNN
    - Distance metrics
    - Curse of dimensionality ⚠️
  - Naive Bayes
    - Independence assumption
    - Laplace smoothing

- **Evaluation Metrics ⭐🔥**
  - Classification
    - Accuracy (imbalanced data ⚠️)
    - Precision, Recall, F1
    - ROC-AUC, PR-AUC
    - Confusion Matrix
  - Regression
    - MAE, RMSE, R²
    - MAPE (zero values ⚠️)
  - Trade-offs
    - Precision vs Recall
    - Threshold selection

## Unsupervised Learning
- **Clustering**
  - K-Means ⭐
    - K selection (Elbow, Silhouette)
    - Initialization sensitivity ⚠️
  - Hierarchical
  - DBSCAN — density-based
  - Gaussian Mixture Models

- **Dimensionality Reduction**
  - PCA ⭐🧠
    - Eigen decomposition
    - Explained variance ratio
    - When NOT to use ⚠️
  - t-SNE — visualization only ⚠️
  - UMAP

- **Anomaly Detection**
  - Isolation Forest
  - One-Class SVM
  - Statistical methods

## Core Concepts ⭐🔥
- **Bias-Variance Tradeoff ⭐🧠**
  - Underfitting vs Overfitting
  - Learning curves
  - Validation strategies

- **Cross-Validation ⭐**
  - K-Fold, Stratified K-Fold
  - Time Series Split ⚠️
  - Leave-One-Out

- **Feature Engineering ⭐🔥**
  - Scaling (Standard vs MinMax)
  - Encoding (One-hot, Target, Ordinal)
  - Binning, Transformations
  - Feature Selection
    - Filter, Wrapper, Embedded

- **Imbalanced Data 🔥**
  - SMOTE, ADASYN
  - Class weights
  - Threshold moving
  - Sampling strategies

- **Model Interpretability**
  - SHAP values ⭐🔥
  - LIME
  - Feature importance (permutation vs impurity)

---

# Deep Learning

## Fundamentals ⭐
- **Neural Network Basics**
  - Forward/Backward Propagation ⭐🧠
  - Activation Functions
    - ReLU (dying ReLU ⚠️)
    - Sigmoid, Tanh (vanishing grad ⚠️)
    - Leaky ReLU, GELU, Swish
  - Loss Functions
    - MSE, Cross-Entropy
    - Custom losses

- **Optimization ⭐🔥**
  - SGD with Momentum
  - Adam / AdamW ⭐
    - Learning rate scheduling
  - Batch Normalization ⭐
  - Layer Normalization
  - Gradient Clipping

- **Regularization 🔥**
  - Dropout ⭐
  - L1/L2 Weight Decay
  - Early Stopping
  - Data Augmentation

## Architectures ⭐
- **CNNs**
  - Conv layers, Pooling
  - ResNet (skip connections) ⭐
  - Transfer Learning ⭐🔥
  - Object Detection: YOLO, R-CNN

- **RNNs / Sequential**
  - LSTM / GRU ⭐
    - Vanishing gradients solved
  - Bidirectional
  - Attention mechanism

- **Transformers ⭐🔥**
  - Self-Attention ⭐🧠
    - Q, K, V matrices
    - Scaled dot-product
  - Multi-Head Attention
  - Positional Encoding
  - Encoder-Decoder architecture

## Training Challenges ⚠️🔥
- Vanishing/Exploding Gradients
- Dead Neurons
- Overfitting small data
- Hyperparameter tuning
  - Grid vs Random vs Bayesian

---

# Statistics & Probability

## Descriptive Statistics ⭐
- Central Tendency: Mean, Median, Mode
- Spread: Variance, Std Dev, IQR
- Distributions
  - Normal, Binomial, Poisson
  - Central Limit Theorem ⭐

## Inferential Statistics ⭐
- Hypothesis Testing
  - Null vs Alternative
  - p-value interpretation ⚠️🧠
  - Type I / Type II errors
  - Power, Effect size
- Confidence Intervals
- A/B Testing ⭐🔥
  - Sample size calculation
  - Multiple testing correction
  - Sequential testing

## Probability ⭐
- Conditional Probability
- Bayes' Theorem ⭐🧠
- Independence vs Mutual Exclusivity
- Expected Value, Variance

## Common Pitfalls ⚠️
- p-hacking
- Confounding variables
- Simpson's Paradox 🧠
- Correlation ≠ Causation

---

# Natural Language Processing

## Traditional NLP ⭐
- **Text Preprocessing**
  - Tokenization
  - Stop words, Stemming, Lemmatization
  - N-grams

- **Text Representations**
  - Bag of Words
  - TF-IDF ⭐
  - Word2Vec / GloVe ⭐
    - CBOW vs Skip-gram
  - FastText — subword info

## Modern NLP 🔥
- **Contextual Embeddings**
  - ELMo
  - BERT ⭐🔥
    - Masked LM
    - Next Sentence Prediction
  - RoBERTa, ALBERT, DistilBERT
  - GPT family

- **NLP Tasks**
  - Classification
  - NER (Named Entity Recognition)
  - POS Tagging
  - Sentiment Analysis
  - Question Answering
  - Summarization
    - Extractive vs Abstractive

- **Tokenization**
  - BPE (Byte Pair Encoding) ⭐
  - WordPiece
  - SentencePiece
  - Unigram

---

# Large Language Models (LLMs)

## Architecture & Training 🔥
- **Decoder-Only Models**
  - GPT architecture ⭐
  - Causal (autoregressive) attention
  - Next token prediction

- **Training Stages**
  - Pre-training ⭐
    - Self-supervised on large corpus
    - Compute requirements 🔥
  - Fine-tuning ⭐🔥
    - Full fine-tuning vs PEFT
    - Catastrophic forgetting ⚠️
  - Alignment
    - RLHF (Reinforcement Learning from Human Feedback) ⭐🧠
    - DPO (Direct Preference Optimization)

- **Scaling Laws**
  - Chinchilla scaling laws 🔥
  - Compute-optimal training

## Inference & Optimization 🔥
- **Efficient Inference**
  - KV-Cache ⭐🧠
  - Quantization ⭐
    - INT8, INT4, GPTQ, AWQ
  - FlashAttention

- **Context Window**
  - Positional interpolation
  - RoPE (Rotary Position Embedding)
  - Long-context challenges ⚠️

## Prompt Engineering ⭐🔥
- Zero-shot vs Few-shot
- Chain-of-Thought (CoT) ⭐
- ReAct (Reasoning + Acting)
- Self-Consistency
- Tree of Thoughts

## LLM Evaluation 🔥
- Perplexity
- BLEU, ROUGE (traditional)
- Human evaluation
- LLM-as-a-Judge ⚠️
- Benchmarks: MMLU, HellaSwag, HumanEval

---

# Retrieval-Augmented Generation (RAG)

## Core Components 🔥
- **Document Processing**
  - Chunking strategies ⭐
    - Fixed-size, Semantic, Recursive
    - Chunk size trade-offs
  - Metadata preservation

- **Embedding Models**
  - Sentence Transformers ⭐
  - E5, BGE models
  - Cross-encoders for reranking

- **Vector Databases ⭐🔥**
  - FAISS, Chroma, Pinecone, Weaviate
  - HNSW indexing
  - Approximate Nearest Neighbors

## RAG Pipeline ⭐
- **Retrieval**
  - Dense retrieval (embeddings)
  - Sparse retrieval (BM25)
  - Hybrid search ⭐
  - Reranking

- **Generation**
  - Context injection
  - Prompt templates
  - Context window limits ⚠️

## Advanced RAG 🔥
- Query rewriting/expansion
- Hypothetical Document Embeddings (HyDE)
- Self-RAG
- Corrective RAG
- Multi-modal RAG

## RAG Evaluation ⭐⚠️
- Retrieval metrics
  - Hit Rate, MRR, NDCG
- Generation metrics
  - Faithfulness, Answer Relevance
  - Context Precision, Recall
- RAGAS framework

---

# Deployment & MLOps

## Model Deployment 🔥
- **Serving Patterns**
  - Real-time (REST/gRPC)
  - Batch inference
  - Streaming

- **Model Formats**
  - ONNX ⭐
  - TensorRT
  - TorchScript

- **Optimization**
  - Model quantization
  - Pruning
  - Knowledge Distillation

## MLOps Pipeline ⭐🔥
- **Experiment Tracking**
  - MLflow, Weights & Biases
  - Reproducibility

- **Feature Store**
  - Online vs Offline features
  - Feature versioning
  - Feast, Tecton

- **Model Registry**
  - Versioning
  - Staging (dev/staging/prod)

- **CI/CD for ML**
  - Automated testing
  - Model promotion
  - A/B testing in production

## Monitoring 🔥⚠️
- **Data Drift**
  - Covariate shift
  - Concept drift
  - Label shift
- **Model Drift**
  - Performance degradation
- **Monitoring Tools**
  - Evidently, WhyLabs

## Infrastructure
- Containerization (Docker)
- Orchestration (Kubernetes)
- Cloud platforms (AWS/GCP/Azure)
- Serverless inference

---

# 🚨 Missing / Often Overlooked Topics

## Critical Pitfalls ⚠️🔥
- **Data Leakage** ⭐⚠️
  - Train-test contamination
  - Target leakage
  - Temporal leakage in time series
  - Preprocessing leakage (fit on test) ⚠️

- **Distribution Shift** 🔥
  - Training-serving skew
  - Covariate shift
  - Concept drift over time
  - Domain adaptation

- **Evaluation Traps** ⚠️
  - Data snooping
  - Multiple comparisons problem
  - Optimism bias in CV

## LLM-Specific Risks 🔥
- **Prompt Injection** ⭐⚠️
  - Direct vs Indirect
  - Jailbreaking techniques
  - Defensive measures

- **Hallucination** ⭐🔥
  - Types: Factual, Faithfulness
  - Mitigation strategies
  - Uncertainty quantification

- **Toxicity & Bias**
  - Output filtering
  - Constitutional AI

## Production Realities 🔥
- **Offline vs Online Evaluation** ⭐⚠️
  - Holdout sets
  - Shadow deployments
  - Interleaving experiments

- **Cold Start Problem**
  - New users/items
  - Warm-up strategies

- **Cost Optimization** 🔥
  - LLM API costs
  - Caching strategies
  - Model routing (small vs large)

## Often Forgotten
- **Ethics & Fairness**
  - Demographic parity
  - Equalized odds
  - Explainability requirements

- **Security**
  - Model inversion attacks
  - Membership inference
  - Model stealing

- **Legal/Compliance**
  - GDPR right to explanation
  - Data retention policies

---

# 🗺️ Suggested Learning Order

## Phase 1: Foundations (Weeks 1-2)
1. Statistics & Probability ⭐
2. Machine Learning Fundamentals ⭐
3. Evaluation Metrics ⭐
4. Cross-Validation & Bias-Variance

## Phase 2: Core ML (Weeks 3-4)
1. Supervised Learning Algorithms ⭐
2. Feature Engineering ⭐
3. Tree-Based Models (XGBoost/LightGBM) 🔥
4. Model Interpretability (SHAP)

## Phase 3: Deep Learning (Weeks 5-6)
1. Neural Network Basics ⭐
2. CNNs & Transfer Learning
3. RNNs & LSTMs
4. Transformers Architecture ⭐🔥

## Phase 4: NLP & LLMs (Weeks 7-8)
1. Traditional NLP & Embeddings
2. BERT & Fine-tuning ⭐
3. GPT & Decoder Models 🔥
4. Prompt Engineering ⭐

## Phase 5: GenAI & RAG (Weeks 9-10)
1. LLM Training & Alignment 🔥
2. RAG Pipeline Design ⭐🔥
3. Vector Databases ⭐
4. LLM Evaluation & Hallucination

## Phase 6: MLOps & Production (Weeks 11-12)
1. Model Deployment Patterns 🔥
2. Monitoring & Drift Detection ⚠️
3. Feature Stores
4. CI/CD for ML

## Phase 7: Interview Prep (Ongoing)
1. Practice coding (LeetCode ML)
2. System design (ML systems)
3. Mock interviews
4. Review "Missing Topics" section ⚠️

---

## Legend
- ⭐ = Frequently asked
- ⚠️ = Common pitfall
- 🔥 = High-impact topic
- 🧠 = Conceptually tricky

---

*Last Updated: 2026-03-23*
*Use this as a living document — mark topics as ✓ when mastered.*

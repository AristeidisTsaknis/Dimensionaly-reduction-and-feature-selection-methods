
```markdown
# dimensionality_reduction

`dimensionality_reduction` is a Python package that provides various implementations of dimensionality reduction and feature selection methods. This package includes popular techniques like PCA, LDA, SVD, LLE, ISOMAP, T-SNE, BORUTA, and more. A key feature of this package is its ability to automatically calculate the optimal number of dimensions for several methods.

## Features

- **Dimensionality Reduction Methods:**
  - Principal Component Analysis (PCA)
  - Linear Discriminant Analysis (LDA)
  - Singular Value Decomposition (SVD)
  - Locally Linear Embedding (LLE)
  - ISOMAP
  - t-Distributed Stochastic Neighbor Embedding (T-SNE)
  - Factor Analysis
  - Kernel PCA
  - Laplacian Eigenmaps
  - Multidimensional Scaling (MDS)
  - Independent Component Analysis (ICA)

- **Feature Selection Methods:**
  - Boruta
  - Ensemble Learning for Feature Selection
  - Kendall Tau Correlation Coefficient
  - Spearman Correlation Coefficient

- **Automated Dimension Calculation:**
  - Automatically calculates the optimal number of dimensions for SVD, PCA, LLE, ISOMAP, Factor Analysis, Kernel PCA, Multidimensional Scaling, Laplacian Eigenmaps, Ensemble Learning for Feature Selection, and LDA.

## Installation

To install the `dimensionality_reduction` package, follow these steps:

1. Download the package from GitHub.
2. Open Command Prompt and navigate to the directory where you downloaded the package.
3. Run the installation command:

```bash
pip install -e .
```

## Usage

Here are some basic examples of how to use the package:

```python
from dimensionality_reduction import PCA, LDA, svd, Locally_Linear_Embedding, Kernel_PCA, isomap, Factor_analysis, Boruta, ensemble_learning_feature_selection, Kendalls_Tau_Correlation, Spearman_Rank_Correlation, G_MDS
from sklearn.datasets import load_iris

# Load sample data
iris = load_iris()
x = iris.data
y = iris.target

# PCA Example
test = PCA()
test.fit_transform(x)

# LDA Example
test = LDA()
test.fit_transform(x, y)

# SVD Example
test = svd()
test.fit_transform(x)

# Locally Linear Embedding Example
test = Locally_Linear_Embedding()
test.fit_transform(x)

# Kernel PCA Example
test = Kernel_PCA()
test.fit_transform(x)

# ISOMAP Example
test = isomap()
test.fit_transform(x)

# Factor Analysis Example
test = Factor_analysis()
test.fit_transform(x)

# Boruta Example
test = Boruta()
results = test.fit(x, y)
print("Boruta test:", results)

# Ensemble Learning Feature Selection Example
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
test = ensemble_learning_feature_selection(estimator=GradientBoostingClassifier())
results = test.fit(X_train, y_train)
print("Ensemble test:", results)

# Kendall Tau Correlation Example
kt = Kendalls_Tau_Correlation()
print(kt.feature_selection(x, y, 0.4))

# Spearman Rank Correlation Example
sp = Spearman_Rank_Correlation()
sp.feature_selection(x, y, 0.4)

# G_MDS Example
g_mds = G_MDS()
values, finished = g_mds.explore_dimensions(x)
g_mds.find_optimal_components(values, finished, x.shape[1])
```

## Requirements

The following libraries are required to use the `dimensionality_reduction` package:

- numpy
- scipy
- scikit-learn
- Kneed
- Matplotlib
- statsmodels

## Author

[Aristeidis Tsaknis]  
[aristeidistsaknis@gmail.com]
```

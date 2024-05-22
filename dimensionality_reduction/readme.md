Thanks for the additional details. Here's the updated README file incorporating the specific requirements and installation instructions:

---

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
2. Open Command Prompt and navigate to your project directory.
3. Run the installation command:

```bash
pip install -e .
```

## Usage

Here are some basic examples of how to use the package:

### PCA Example

```python
from dimensionality_reduction import PCA

# Initialize PCA
pca = PCA()

# Fit and transform the data
reduced_data = pca.fit_transform(data,num_of_dimensions)
```

### LDA Example

```python
from dimensionality_reduction import LDA

# Initialize LDA
lda = LDA()

# Fit and transform the data
reduced_data = lda.fit_transform(data, num_of_dimensions, labels)
```

### Automated Dimension Calculation

```python
from dimensionality_reduction import PCA

# Initialize PCA 
pca = PCA()

# Fit and transform the data with automated dimension calculation
reduced_data = pca.fit_transform(data)
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


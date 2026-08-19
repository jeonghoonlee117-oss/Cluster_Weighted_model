# Polynomial Gaussian Cluster Weighted Model (cwm_gau_poly2)

## Overview
This implementation provides a **Univariate 2nd-Order Polynomial Gaussian Cluster Weighted Model** using the EM algorithm for regression modeling with multiple clusters.

---

## Required Libraries

```r
if (!requireNamespace("cluster", quietly = TRUE)) install.packages("cluster")
library(cluster)
```

---

## Main Function: Polynomial Cluster Weighted Model

### `cwm_gau_poly2()`

Fits a cluster weighted model with 2nd-order polynomial regression.

```r
cwm_gau_poly2 <- function(x, y, k = 1:5,
                          init_type = c("kmeans", "kmedoids", "kmedians"),
                          criterion = c("BIC", "AIC"),
                          max_iter = 1000, tol = 1e-6)
```

#### Function Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `x` | Univariate predictor (numeric vector) | - |
| `y` | Response variable | - |
| `k` | Number of clusters to evaluate | 1:5 |
| `init_type` | Initialization method | "kmeans" |
| `criterion` | Model selection criterion | "BIC" |
| `max_iter` | Maximum EM iterations | 1000 |
| `tol` | Convergence tolerance | 1e-6 |

#### Function Logic

**1. Input Validation**
- Converts x and y to numeric vectors
- Validates that lengths match
- Prepares data for clustering

**2. Initialization**
- Initializes clusters using kmeans, kmedoids, or kmedians
- Fits global 2nd-order polynomial regression
- Sets initial cluster-specific parameters

**3. EM Algorithm**

**E-Step:**
- Computes input density: p(x) ~ N(μ_m, σ²_x,m)
- Computes conditional output density: p(y|x) ~ N(β_0 + β_1·x + β_2·x², σ²_y,m)
- Calculates posterior probabilities using mixture weights

**M-Step:**
- Updates mixing proportions: w_m
- Updates input distribution parameters (μ_m, σ²_x,m)
- Computes S matrix (Hankel form) and T vector
- Solves for polynomial coefficients: **β = S⁻¹ · T**
- Updates output variance: σ²_y,m

**4. Model Selection**
- Calculates BIC/AIC for each k value
- Selects best model based on criterion

#### Return Value

Returns an object of class `"cwm_gau_poly2"` containing:

```r
list(
  best_M           = optimal number of clusters,
  criterion        = criterion used,
  comparison_table = data frame with M, LogLik, df, BIC, AIC,
  cluster          = cluster assignments,
  posterior        = posterior probabilities,
  w                = mixing proportions,
  mu_x             = cluster means for x,
  sigma2_x         = cluster variances for x,
  Beta             = polynomial coefficients (3 × M) - [β₀, β₁, β₂]ᵀ,
  sigma2_y         = cluster variances for y,
  loglik           = log-likelihood,
  BIC              = BIC value,
  AIC              = AIC value,
  models           = list of all fitted models
)
```

---

## Key Components

### Regression Model (Per Cluster)
$$y_m = \beta_{0,m} + \beta_{1,m} x + \beta_{2,m} x^2 + \epsilon_m$$

where $\epsilon_m \sim N(0, \sigma^2_{y,m})$

### S Matrix (Hankel Form)
$$S = \begin{pmatrix} S_0 & S_1 & S_2 \\ S_1 & S_2 & S_3 \\ S_2 & S_3 & S_4 \end{pmatrix}$$

where $S_j = \sum_i \alpha_{im} x_i^j$

### T Vector
$$T = \begin{pmatrix} T_0 \\ T_1 \\ T_2 \end{pmatrix}$$

where $T_j = \sum_i \alpha_{im} x_i^j y_i$

### Coefficient Estimation
$$\boldsymbol{\beta}_m = (S + \lambda I)^{-1} T$$

---

## Usage Example

```r
# Load the function
source("cwm_em_poly_r_code.R")

# Generate sample data
set.seed(123)
n <- 300
x <- seq(-5, 5, length.out = n)
y <- 2 + 1.5*x - 0.3*x^2 + rnorm(n, sd = 1)

# Fit the model
result <- cwm_gau_poly2(x, y, k = 1:4, init_type = "kmeans", criterion = "BIC")

# Explore results
print(result$comparison_table)        # Compare models
print(result$best_M)                  # Optimal clusters
print(result$Beta)                    # Polynomial coefficients
print(result$cluster)                 # Cluster assignments

# Visualization (pseudo-code)
# plot(x, y, col = result$cluster, main = "Cluster Weighted Polynomial Regression")
# lines(x, result$Beta[1,1] + result$Beta[2,1]*x + result$Beta[3,1]*x^2)
```

---

## Degrees of Freedom

$$df = (M-1) + M \times (1 + 1 + 3 + 1) = 6M - 1$$

Components:
- Mixing proportions: M - 1
- Input mean (μ): M
- Input variance (σ²_x): M
- Polynomial coefficients (β₀, β₁, β₂): 3M
- Output variance (σ²_y): M

---

## Information Criteria

**BIC (Bayesian Information Criterion):**
$$BIC = -2 \ell + df \times \log(N)$$

**AIC (Akaike Information Criterion):**
$$AIC = -2 \ell + 2 \times df$$

---

## Key Features

✓ **Polynomial Regression**: 2nd-order polynomial for flexible curve fitting  
✓ **Mixture Modeling**: Handles heterogeneous clusters with different regression lines  
✓ **Flexible Initialization**: Three clustering methods (kmeans, kmedoids, kmedians)  
✓ **Model Selection**: Automatic BIC/AIC comparison  
✓ **Posterior Probabilities**: Soft cluster assignments available  
✓ **Numerical Stability**: Regularization of S matrix with λ·I

---

## Convergence Criteria

- Convergence tolerance: `tol = 1e-6` (default)
- Convergence check: $|\ell_{new} - \ell_{old}| < tol$
- Maximum iterations: `max_iter = 1000` (default)

---

## Notes

- Requires the `cluster` package for PAM clustering
- Handles cases with few observations per cluster
- S matrix regularization (1e-8) prevents singularity
- Global polynomial fit provides stable initialization
- Input (x) and output (y) modeled as joint Gaussian mixture

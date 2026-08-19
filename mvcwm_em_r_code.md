# Linear Gaussian Cluster Weighted Model (cwm)

## Overview
This implementation provides a **Multivariate Linear Gaussian Cluster Weighted Model** using the EM algorithm for regression modeling with multiple clusters and heteroscedastic errors.

---

## Required Libraries

```r
if (!requireNamespace("cluster", quietly = TRUE)) install.packages("cluster")
library(cluster)
```

---

## Main Function: Linear Gaussian Cluster Weighted Model

### `cwm()`

Fits a cluster weighted model with linear regression and Gaussian errors.

```r
cwm <- function(X, y, k = 1:5,
                init_type = c("kmeans", "kmedoids", "kmedians"),
                criterion = c("BIC", "AIC"),
                max_iter = 1000, tol = 1e-6)
```

#### Function Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `X` | Predictor matrix (n × p) | - |
| `y` | Continuous response variable | - |
| `k` | Number of clusters to evaluate | 1:5 |
| `init_type` | Initialization method | "kmeans" |
| `criterion` | Model selection criterion | "BIC" |
| `max_iter` | Maximum EM iterations | 1000 |
| `tol` | Convergence tolerance | 1e-6 |

#### Function Logic

**1. Input Preparation**
- Converts X to matrix format
- Creates design matrix with intercept
- Initializes parameter storage arrays

**2. Initialization**
- Initializes clusters using kmeans, kmedoids, or kmedians
- Fits global linear regression
- Sets initial cluster-specific parameters

**3. EM Algorithm**

**E-Step:**
- Computes multivariate Gaussian density for X: p(x) ~ N(μ_m, Σ_m)
- Computes conditional density for y|x: p(y|x) ~ N(ŷ_m, σ²_y,m)
  - where ŷ_m = β₀ + β₁x₁ + ... + β_px_p
- Calculates posterior probabilities (responsibilities): α_ij

**M-Step:**
- Updates mixing proportions: w_m = Σᵢ α_im / N
- Updates input distribution (Gaussian):
  - μ_m = Σᵢ α_im · x_i / Σᵢ α_im
  - Σ_m = [Σᵢ α_im (x_i - μ_m)(x_i - μ_m)ᵀ / Σᵢ α_im] + λI
- Updates regression parameters via weighted least squares:
  - **β_m = (X^T W X)⁻¹ X^T W y**
  - where W = diag(α_im)
- Updates error variance:
  - σ²_y,m = Σᵢ α_im (y_i - ŷ_i)² / Σᵢ α_im

**4. Model Selection**
- Calculates BIC/AIC for each k value
- Selects best model based on criterion

#### Return Value

Returns an object of class `"cwm"` containing:

```r
list(
  best_M           = optimal number of clusters,
  criterion        = criterion used,
  comparison_table = data frame with M, LogLik, df, BIC, AIC,
  cluster          = hard cluster assignments,
  posterior        = posterior probabilities (N × M),
  w                = mixing proportions (M),
  mu_x             = cluster means (p × M),
  Sigma_x          = cluster covariances (list of M matrices),
  Beta             = regression coefficients (p+1 × M),
  sigma_y          = error standard deviations (M),
  loglik           = final log-likelihood,
  BIC              = BIC value,
  AIC              = AIC value,
  models           = list of all fitted models
)
```

---

## Key Components

### Model Specification

**Mixture Distribution:**
$$p(x, y) = \sum_{m=1}^{M} w_m p(x | \mu_m, \Sigma_m) p(y | x, \beta_m, \sigma^2_{y,m})$$

**Cluster-Specific Regression:**
$$y_m = \beta_{0,m} + \beta_{1,m}x_1 + \cdots + \beta_{p,m}x_p + \epsilon_m$$

where $\epsilon_m \sim N(0, \sigma^2_{y,m})$

### Weighted Least Squares Estimation

For cluster m:
$$\hat{\beta}_m = (X^T W_m X + \lambda I)^{-1} X^T W_m y$$

where $W_m = \text{diag}(\alpha_{1m}, \alpha_{2m}, \ldots, \alpha_{Nm})$ (posterior weights)

### Covariance Update

$$\Sigma_m = \frac{\sum_i \alpha_{im} (x_i - \mu_m)(x_i - \mu_m)^T}{\sum_i \alpha_{im}} + \lambda I$$

Regularization λ = 1e-6 ensures positive-definiteness

### Log-Likelihood

$$\ell = \sum_i \log\left(\sum_m w_m \phi(x_i|\mu_m, \Sigma_m) \phi(y_i|x_i, \beta_m, \sigma^2_{y,m})\right)$$

---

## Degrees of Freedom

$$df = (M-1) + M \times [p + \frac{p(p+1)}{2} + (p+1) + 1]$$

$$df = (M-1) + M \times [2p + \frac{p(p+1)}{2} + 2]$$

Components:
- Mixing proportions: M - 1
- Input means (μ): M × p
- Input covariance (Σ): M × p(p+1)/2
- Regression coefficients (β): M × (p+1)
- Error variance (σ²_y): M

---

## Information Criteria

**BIC (Bayesian Information Criterion):**
$$BIC = -2\ell + df \times \log(N)$$

Penalizes model complexity; larger penalty for sample size

**AIC (Akaike Information Criterion):**
$$AIC = -2\ell + 2 \times df$$

Less stringent penalty; useful for prediction-focused selection

---

## Usage Example

```r
# Load the function
source("mvcwm_em_r_code.R")

# Generate sample data with 2 clusters
set.seed(123)
n <- 300
X <- matrix(rnorm(n * 2), ncol = 2)
y <- ifelse(runif(n) > 0.5,
            1 + 2*X[, 1] - X[, 2] + rnorm(n, sd = 0.5),
            -1 - X[, 1] + 1.5*X[, 2] + rnorm(n, sd = 0.5))

# Fit the model
result <- cwm(X, y, k = 1:5, init_type = "kmeans", criterion = "BIC")

# Explore results
print(result$comparison_table)        # Model comparison
print(result$best_M)                  # Optimal clusters
print(result$cluster)                 # Cluster assignments
print(result$Beta)                    # Regression coefficients per cluster
print(result$sigma_y)                 # Error std dev per cluster

# Predictions for new data
new_X <- matrix(c(0.5, -0.3, 1.2, 0.8), nrow = 2, ncol = 2)
for (m in 1:result$best_M) {
  pred <- cbind(1, new_X) %*% result$Beta[, m]
  print(paste("Cluster", m, "predictions:", paste(pred, collapse = ", ")))
}
```

---

## Key Features

✓ **Linear Regression**: Standard linear model per cluster  
✓ **Mixture Modeling**: Handles heterogeneous clusters with different regression lines  
✓ **Flexible Initialization**: Three clustering methods (kmeans, kmedoids, kmedians)  
✓ **Model Selection**: Automatic BIC/AIC comparison across k values  
✓ **Posterior Probabilities**: Soft cluster assignments available  
✓ **Weighted Estimation**: Each observation weighted by posterior probability  
✓ **Cluster-Specific Variance**: Heteroscedastic error terms across clusters  

---

## Convergence Criteria

- Convergence tolerance: `tol = 1e-6` (default)
- Convergence check: $|\ell_{new} - \ell_{old}| < tol$
- Maximum EM iterations: `max_iter = 1000` (default)
- Hard assignments: argmax(α_ij) for each observation

---

## Numerical Stability

- **Covariance regularization**: $\Sigma_m + \lambda I$ (λ = 1e-6)
- **Design matrix regularization**: Weighted normal equations with λ·I (λ = 1e-8)
- **Density floor**: Ensures $p(x_i, y_i) \geq 1e^{-300}$
- **Weight floor**: Ensures $\sum \alpha_i \geq 1e^{-12}$

---

## Notes

- Requires the `cluster` package for PAM clustering
- Handles small cluster sizes with fallback to global estimates
- Covariance estimation uses sample covariance + regularization
- Regression coefficients initialized from cluster-specific OLS fits
- Posterior probabilities (soft assignments) in `$posterior` matrix
- Hard assignments via `$cluster` (argmax of posterior)

---

## Comparison with Other Methods

| Feature | CWM | Standard Mixture | K-means | Linear Regression |
|---------|-----|------------------|--------|-------------------|
| Soft clusters | ✓ | ✓ | ✗ | ✗ |
| Input distribution | ✓ | ✓ | ✓ | ✗ |
| Output modeling | ✓ | ✗ | ✗ | ✓ |
| Multiple regression lines | ✓ | ✗ | ✗ | ✗ |
| Model selection (BIC/AIC) | ✓ | ✓ | ✗ | ✓ |

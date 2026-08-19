# Poisson Cluster Weighted Model (mvcwm_poi)

## Overview
This implementation provides a **Multivariate Poisson Cluster Weighted Model** using the EM algorithm with BFGS optimization for weighted Poisson regression coefficients.

---

## Required Libraries

```r
if (!requireNamespace("cluster", quietly = TRUE)) install.packages("cluster")
library(cluster)
```

---

## Helper Function: Weighted Poisson Regression Coefficient Update

### `update_poisson_beta_optim()`

Updates Poisson regression coefficients using BFGS optimization.

```r
update_poisson_beta_optim <- function(X_design, y, alpha_i, beta_start)
```

#### Function Details

**Objective Function (Negative Weighted Log-Likelihood):**
```r
obj_fn <- function(beta) {
  eta <- as.vector(X_design %*% beta)
  eta <- pmin(pmax(eta, -700), 700)  # Clip for numerical stability
  lambda <- exp(eta)
  neg_loglik <- sum(alpha_i * (lambda - y * eta))
  return(neg_loglik)
}
```

**Gradient Function:**
```r
grad_fn <- function(beta) {
  eta <- as.vector(X_design %*% beta)
  eta <- pmin(pmax(eta, -700), 700)
  lambda <- exp(eta)
  grad <- crossprod(X_design, alpha_i * (lambda - y))
  return(as.vector(grad))
}
```

**Parameters:**
- `X_design`: Design matrix (with intercept column)
- `y`: Count response variable (non-negative integers)
- `alpha_i`: Weights from E-step
- `beta_start`: Initial coefficient values

**Returns:** 
- Optimized Poisson regression coefficients

---

## Main Function: Multivariate Poisson Cluster Weighted Model

### `mvcwm_poi()`

Fits a cluster weighted model with Poisson regression.

```r
mvcwm_poi <- function(X, y, k = 1:5,
                      init_type = c("kmeans", "kmedoids", "kmedians"),
                      criterion = c("BIC", "AIC"),
                      max_iter = 1000, tol = 1e-6)
```

#### Function Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `X` | Predictor matrix (n × p) | - |
| `y` | Count response (non-negative integers) | - |
| `k` | Number of clusters to evaluate | 1:5 |
| `init_type` | Initialization method | "kmeans" |
| `criterion` | Model selection criterion | "BIC" |
| `max_iter` | Maximum EM iterations | 1000 |
| `tol` | Convergence tolerance | 1e-6 |

#### Function Logic

**1. Input Validation**
- Converts X to matrix format
- Validates count response: y ≥ 0 and y ∈ ℤ
- Creates design matrix with intercept
- Initializes cluster parameter arrays

**2. Initialization**
- Initializes clusters using kmeans, kmedoids, or kmedians
- Log-transforms response for clustering (log1p)
- Fits global Poisson GLM via `glm(family = poisson(link = "log"))`
- Sets initial cluster-specific parameters

**3. EM Algorithm**

**E-Step:**
- Computes multivariate Gaussian density: p(x) ~ N(μ_m, Σ_m)
- Computes Poisson likelihood: p(y|x) ~ Poisson(λ_m)
  - where λ_m = exp(η) = exp(β₀ + β₁x₁ + ... + β_px_p)
- Calculates posterior probabilities: α_ij

**M-Step:**
- Updates mixing proportions: w_i
- Updates Gaussian parameters (μ_m, Σ_m)
- Updates Poisson regression coefficients via BFGS
  - Maximizes: Q_m = Σᵢ α_im log[p(x_i, y_i|θ_m)]

**4. Model Selection**
- Calculates BIC/AIC for each k value
- Selects best model based on criterion

#### Return Value

Returns an object of class `"mvcwm_poi"` containing:

```r
list(
  best_M           = optimal number of clusters,
  criterion        = criterion used,
  comparison_table = data frame with M, LogLik, df, BIC, AIC,
  cluster          = cluster assignments,
  posterior        = posterior probabilities,
  w                = mixing proportions,
  mu_x             = cluster means (p × M),
  Sigma_x          = cluster covariances (list of M matrices),
  Beta             = Poisson coefficients (p+1 × M),
  loglik           = log-likelihood,
  BIC              = BIC value,
  AIC              = AIC value,
  models           = list of all fitted models
)
```

---

## Key Components

### Poisson Regression Model (Per Cluster)
$$E[y_m|x] = \lambda_m = \exp(\beta_{0,m} + \beta_{1,m}x_1 + \cdots + \beta_{p,m}x_p)$$

### Log-Likelihood
$$\ell_m = \sum_i \alpha_{im} [y_i \eta_{im} - \lambda_{im}]$$

where $\eta_{im} = \log(\lambda_{im})$

### Weighted Poisson Estimation
Solves via BFGS:
$$\hat{\beta}_m = \arg\max_{\beta} \sum_i \alpha_{im} [y_i \eta_i - \exp(\eta_i)]$$

### Gradient
$$\nabla Q = X^T(\alpha_i (\lambda - y))$$

where $\lambda = \exp(X\beta)$

---

## Usage Example

```r
# Load the function
source("mvcwm_em_poi_r_code.R")

# Generate sample count data
set.seed(123)
n <- 200
X <- matrix(rnorm(n * 3), ncol = 3)
y <- rpois(n, lambda = 5)

# Fit the model
result <- mvcwm_poi(X, y, k = 1:4, init_type = "kmeans", criterion = "BIC")

# Explore results
print(result$comparison_table)        # Compare models
print(result$best_M)                  # Optimal clusters
print(result$cluster)                 # Cluster assignments
print(result$Beta)                    # Poisson regression coefficients
print(result$posterior[1:5, ])        # First 5 posterior probabilities

# Prediction example (pseudo-code)
# new_X <- matrix(c(0.5, -0.3, 1.2), nrow = 1)
# eta <- c(1, new_X) %*% result$Beta[, best_cluster]
# predicted_lambda <- exp(eta)
```

---

## Degrees of Freedom

$$df = (M-1) + M \times [2p + \frac{p(p+1)}{2} + 1]$$

Components:
- Mixing proportions: M - 1
- Input means (μ): M × p
- Input covariance (Σ): M × p(p+1)/2
- Poisson coefficients (β): M × (p+1)

---

## Information Criteria

**BIC (Bayesian Information Criterion):**
$$BIC = -2\ell + df \times \log(N)$$

**AIC (Akaike Information Criterion):**
$$AIC = -2\ell + 2 \times df$$

---

## Key Features

✓ **Count Data Modeling**: Poisson regression for count/frequency data  
✓ **Mixture Modeling**: Handles heterogeneous clusters with different mean structures  
✓ **Log-Link Function**: Natural Poisson parameter link ensures λ > 0  
✓ **Flexible Initialization**: Three clustering methods  
✓ **Model Selection**: Automatic BIC/AIC comparison  
✓ **Posterior Probabilities**: Soft cluster assignments available  
✓ **Numerical Stability**: Clipping of η to [-700, 700]

---

## Convergence Criteria

- Convergence tolerance: `tol = 1e-6` (default)
- Convergence check: $|\ell_{new} - \ell_{old}| < tol$
- Maximum EM iterations: `max_iter = 1000` (default)
- BFGS max iterations: 200
- BFGS relative tolerance: 1e-8

---

## Numerical Stability Features

- **Eta clipping**: $\eta \in [-700, 700]$ prevents overflow/underflow
- **Covariance regularization**: $\Sigma_m + \lambda I$ (λ = 1e-6)
- **Density floor**: $p(x_i, y_i) \geq 1e^{-300}$ prevents log(0)
- **Weight floor**: $\sum \alpha_i \geq 1e^{-12}$ prevents division by zero

---

## Notes

- Requires the `cluster` package for PAM clustering
- Handles non-count (continuous) input X normally
- Output y must be non-negative integers
- Automatic initialization from global Poisson GLM
- Missing coefficients replaced with global estimates

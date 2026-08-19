# Binary Logistic Cluster Weighted Model (mvcwm_bin)

## Overview
This implementation provides a **Multivariate Binary Logistic Cluster Weighted Model** using the EM algorithm with BFGS optimization for weighted logistic regression coefficients.

---

## Required Libraries

```r
if (!requireNamespace("cluster", quietly = TRUE)) install.packages("cluster")
library(cluster)
```

---

## Helper Function: Weighted Logistic Regression Coefficient Update

### `update_logistic_beta_optim()`

Updates logistic regression coefficients using BFGS optimization.

```r
update_logistic_beta_optim <- function(X_design, y, alpha_i, beta_start) {
  
  # Objective function: Negative weighted log-likelihood 
  # (with log1pexp for numerical stability)
  obj_fn <- function(beta) {
    eta <- as.vector(X_design %*% beta)
    eta <- pmin(pmax(eta, -700), 700)
    log1pexp_eta <- ifelse(eta > 0, eta + log1p(exp(-eta)), log1p(exp(eta)))
    neg_loglik <- sum(alpha_i * (log1pexp_eta - y * eta))
    return(neg_loglik)
  }
  
  # Gradient function: t(X) %*% (alpha * (pi - y))
  grad_fn <- function(beta) {
    eta <- as.vector(X_design %*% beta)
    pi_hat <- 1 / (1 + exp(-pmin(pmax(eta, -700), 700)))
    grad <- crossprod(X_design, alpha_i * (pi_hat - y))
    return(as.vector(grad))
  }
  
  opt_res <- optim(
    par     = beta_start,
    fn      = obj_fn,
    gr      = grad_fn,
    method  = "BFGS",
    control = list(maxit = 200, reltol = 1e-8)
  )
  
  return(opt_res$par)
}
```

**Parameters:**
- `X_design`: Design matrix (with intercept column)
- `y`: Binary response variable (0 or 1)
- `alpha_i`: Weights from E-step
- `beta_start`: Initial coefficient values

**Returns:** 
- Optimized regression coefficients

---

## Main Function: Multivariate Logistic Cluster Weighted Model

### `mvcwm_bin()`

Fits a cluster weighted model with binary logistic regression.

```r
mvcwm_bin <- function(X, y, k = 1:5,
                      init_type = c("kmeans", "kmedoids", "kmedians"),
                      criterion = c("BIC", "AIC"),
                      max_iter = 1000, tol = 1e-6)
```

#### Function Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `X` | Predictor matrix (n × p) | - |
| `y` | Binary response (0 or 1) | - |
| `k` | Number of clusters to evaluate | 1:5 |
| `init_type` | Initialization method | "kmeans" |
| `criterion` | Model selection criterion | "BIC" |
| `max_iter` | Maximum EM iterations | 1000 |
| `tol` | Convergence tolerance | 1e-6 |

#### Function Logic

**1. Input Validation**
- Converts X to matrix format
- Validates binary response: y ∈ {0, 1}
- Creates design matrix with intercept

**2. Initialization**
- Initializes clusters using kmeans, kmedoids, or kmedians
- Fits global logistic GLM
- Sets initial cluster-specific parameters

**3. EM Algorithm**

**E-Step:**
- Computes multivariate Gaussian density: p(x)
- Computes logistic likelihood: p(y|x)
- Calculates posterior probabilities: α_ij = (w_i × p(x) × p(y|x)) / Σ

**M-Step:**
- Updates mixing proportions: w_i
- Updates Gaussian parameters (μ, Σ)
- Updates logistic regression coefficients via BFGS

**4. Model Selection**
- Calculates BIC/AIC for each k value
- Selects best model based on criterion

#### Return Value

Returns an object of class `"mvcwm_bin"` containing:

```r
list(
  best_M           = optimal number of clusters,
  criterion        = criterion used,
  comparison_table = data frame with M, LogLik, df, BIC, AIC,
  cluster          = cluster assignments,
  posterior        = posterior probabilities,
  w                = mixing proportions,
  mu_x             = cluster means,
  Sigma_x          = cluster covariances,
  Beta             = logistic coefficients (p+1 × M),
  loglik           = log-likelihood,
  BIC              = BIC value,
  AIC              = AIC value,
  models           = list of all fitted models
)
```

---

## Usage Example

```r
# Load the function
source("cwm_em_bin_r_code.R")

# Generate sample data
set.seed(123)
n <- 200
X <- matrix(rnorm(n * 3), ncol = 3)
y <- rbinom(n, size = 1, prob = 0.5)

# Fit the model
result <- mvcwm_bin(X, y, k = 1:4, init_type = "kmeans", criterion = "BIC")

# Explore results
print(result$comparison_table)        # Compare models
print(result$best_M)                  # Optimal clusters
print(result$cluster)                 # Cluster assignments
print(result$Beta)                    # Logistic coefficients
print(result$posterior[1:5, ])        # First 5 posterior probabilities
```

---

## Key Features

✓ **Numerical Stability**: Log1pexp transformation prevents overflow/underflow  
✓ **Flexible Initialization**: Three clustering methods for initialization  
✓ **Model Selection**: Automatic BIC/AIC comparison  
✓ **Robust Estimation**: BFGS optimization for coefficient updates  
✓ **Posterior Probabilities**: Soft cluster assignments available  

---

## Mathematical Details

### Log-Likelihood
$$\ell = \sum_{i=1}^{n} \log\left(\sum_{m=1}^{M} w_m p(x_i|\mu_m, \Sigma_m) p(y_i|x_i, \beta_m)\right)$$

### Gaussian Component
$$p(x|\mu_m, \Sigma_m) = (2\pi)^{-p/2}|\Sigma_m|^{-1/2} \exp\left(-\frac{1}{2}(x-\mu_m)^T\Sigma_m^{-1}(x-\mu_m)\right)$$

### Logistic Component
$$p(y|x, \beta_m) = \pi_m^y (1-\pi_m)^{1-y}, \quad \pi_m = \frac{1}{1+e^{-\eta}}$$

where $\eta = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p$

---

## Notes

- Requires the `cluster` package for PAM clustering
- Handles near-singular covariance matrices with regularization (1e-6)
- Clips log-odds to [-700, 700] for numerical stability
- Automatically handles missing coefficients from GLM fitting

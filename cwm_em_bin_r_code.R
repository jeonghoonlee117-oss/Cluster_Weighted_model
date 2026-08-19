if (!requireNamespace("cluster", quietly = TRUE)) install.packages("cluster")
library(cluster)

# ==============================================================================
# 1. Helper: optim(BFGS) 기반 가중 로지스틱 회귀계수 갱신 함수
# ==============================================================================
update_logistic_beta_optim <- function(X_design, y, alpha_i, beta_start) {
  
  # 목적 함수: 음의 가중 로그 우도 (수치적 안정성을 위한 log1pexp 처리)
  obj_fn <- function(beta) {
    eta <- as.vector(X_design %*% beta)
    eta <- pmin(pmax(eta, -700), 700)
    log1pexp_eta <- ifelse(eta > 0, eta + log1p(exp(-eta)), log1p(exp(eta)))
    neg_loglik <- sum(alpha_i * (log1pexp_eta - y * eta))
    return(neg_loglik)
  }
  
  # 기울기 함수: t(X) %*% (alpha * (pi - y))
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

# ==============================================================================
# 2. 메인 함수: 다변량 로지스틱 군집 가중 모델 (mvcwm_bin)
# ==============================================================================
mvcwm_bin <- function(X, y, k = 1:5,
                      init_type = c("kmeans", "kmedoids", "kmedians"),
                      criterion = c("BIC", "AIC"),
                      max_iter = 1000, tol = 1e-6) {
  
  init_type <- match.arg(init_type)
  criterion <- match.arg(criterion)
  
  X_mat <- as.matrix(X)
  N <- nrow(X_mat)
  D <- ncol(X_mat)
  y <- as.numeric(y)
  
  # 이진 분류 유효성 검증
  if (any(!y %in% c(0, 1))) {
    stop("로지스틱(베르누이) 모델의 반응 변수 y는 0 또는 1이어야 합니다.")
  }
  
  X_design <- cbind(1, X_mat)
  colnames(X_design) <- c("(Intercept)", if (!is.null(colnames(X))) colnames(X) else paste0("x", 1:D))
  
  fit_single_mvcwm_bin <- function(M) {
    
    if (M == 1) {
      init_cluster <- rep(1, N)
    } else {
      XY_mat <- cbind(scale(X_mat), scale(y))
      init_cluster <- switch(
        init_type, 
        kmeans   = kmeans(XY_mat, centers = M, nstart = 20)$cluster,
        kmedoids = pam(XY_mat, k = M, metric = "euclidean")$clustering,
        kmedians = pam(XY_mat, k = M, metric = "manhattan")$clustering
      )
    }
    
    w <- numeric(M)
    mu_x <- matrix(0, nrow = D, ncol = M)
    Sigma_x <- vector("list", M)
    Beta <- matrix(0, nrow = D + 1, ncol = M)
    
    # 전체 초기 Beta (로지스틱 GLM 적용)
    global_fit <- suppressWarnings(glm(y ~ X_mat, family = binomial(link = "logit")))
    global_beta <- coef(global_fit)
    global_beta[is.na(global_beta)] <- 0
    
    for (i in 1:M) {
      idx <- which(init_cluster == i)
      n_i <- length(idx)
      w[i] <- n_i / N
      
      if (n_i <= 1) {
        mu_x[, i] <- if (n_i == 1) X_mat[idx, ] else colMeans(X_mat)
        Sigma_x[[i]] <- diag(1, D)
      } else {
        mu_x[, i] <- colMeans(X_mat[idx, , drop = FALSE])
        Sigma_x[[i]] <- cov(X_mat[idx, , drop = FALSE]) + diag(1e-6, D)
      }
      
      if (n_i <= (D + 1)) {
        Beta[, i] <- global_beta
      } else {
        local_fit <- tryCatch(
          suppressWarnings(glm(y[idx] ~ X_mat[idx, , drop = FALSE], family = binomial(link = "logit"))),
          error = function(e) NULL
        )
        if (!is.null(local_fit) && !any(is.na(coef(local_fit)))) {
          Beta[, i] <- coef(local_fit)
        } else {
          Beta[, i] <- global_beta
        }
      }
    }
    
    loglik_history <- numeric()
    loglik_old <- -Inf
    
    for (iter in 1:max_iter) {
      
      # [E-Step]
      density_mat <- matrix(0, nrow = N, ncol = M)
      
      for (i in 1:M) {
        mu_i <- mu_x[, i]
        Sigma_i <- Sigma_x[[i]]
        mah_dist <- mahalanobis(X_mat, center = mu_i, cov = Sigma_i)
        
        log_det <- as.numeric(determinant(Sigma_i, logarithm = TRUE)$modulus)
        log_p_x <- -0.5 * (D * log(2 * pi) + log_det + mah_dist)
        p_x <- exp(log_p_x)
        
        # 로지스틱 확률 계산
        eta <- as.vector(X_design %*% Beta[, i])
        eta <- pmin(pmax(eta, -700), 700)
        pi_hat <- 1 / (1 + exp(-eta))
        p_y_given_x <- dbinom(y, size = 1, prob = pmax(pmin(pi_hat, 1 - 1e-10), 1e-10))
        
        density_mat[, i] <- w[i] * p_x * p_y_given_x
      }
      
      total_density <- pmax(rowSums(density_mat), 1e-300)
      alpha <- density_mat / total_density
      
      loglik_new <- sum(log(total_density))
      loglik_history <- c(loglik_history, loglik_new)
      
      if (abs(loglik_new - loglik_old) < tol) break
      loglik_old <- loglik_new
      
      # [M-Step]
      for (i in 1:M) {
        a <- alpha[, i]
        sum_a <- max(sum(a), 1e-12)
        
        w[i] <- sum_a / N
        
        mu_x[, i] <- colSums(a * X_mat) / sum_a
        X_centered <- sweep(X_mat, 2, mu_x[, i], "-")
        Sigma_x[[i]] <- (crossprod(X_centered, X_centered * a) / sum_a) + diag(1e-6, D)
        
        # 로지스틱 Beta 갱신 함수 연결
        Beta[, i] <- update_logistic_beta_optim(
          X_design   = X_design,
          y          = y,
          alpha_i    = a, 
          beta_start = Beta[, i]
        )
      }
    }
    
    # 자유도 (포아송과 동일하게 분산 파라미터가 없으므로 동일한 공식 적용)
    df <- (M - 1) + M * (2 * D + (D * (D + 1)) / 2 + 1)
    bic_val <- -2 * loglik_new + df * log(N)
    aic_val <- -2 * loglik_new + 2 * df
    
    return(list(
      M = M, D = D, N = N, df = df,
      iterations = iter,
      loglik_history = loglik_history,
      loglik = loglik_new,
      BIC = bic_val, 
      AIC = aic_val,
      w = w, mu_x = mu_x, Sigma_x = Sigma_x,
      Beta = Beta,
      alpha = alpha,
      cluster = apply(alpha, 1, which.max)
    ))
  }
  
  k_candidates <- sort(unique(as.integer(k)))
  num_k <- length(k_candidates)
  
  models <- vector("list", num_k)
  names(models) <- paste0("M=", k_candidates)
  
  comparison_table <- data.frame(
    M      = k_candidates,
    LogLik = numeric(num_k),
    df     = numeric(num_k),
    BIC    = numeric(num_k),
    AIC    = numeric(num_k)
  )
  
  for (idx in seq_along(k_candidates)) {
    m_val <- k_candidates[idx]
    fit <- fit_single_mvcwm_bin(m_val)
    
    models[[idx]] <- fit
    comparison_table$LogLik[idx] <- fit$loglik
    comparison_table$df[idx]     <- fit$df
    comparison_table$BIC[idx]    <- fit$BIC
    comparison_table$AIC[idx]    <- fit$AIC
  }
  
  crit_values <- if (criterion == "BIC") comparison_table$BIC else comparison_table$AIC
  best_idx <- which.min(crit_values)
  best_model <- models[[best_idx]]
  
  res <- list(
    best_M           = best_model$M,
    criterion        = criterion,
    comparison_table = comparison_table,
    cluster          = best_model$cluster,
    posterior        = best_model$alpha,
    w                = best_model$w,
    mu_x             = best_model$mu_x,
    Sigma_x          = best_model$Sigma_x,
    Beta             = best_model$Beta,
    loglik           = best_model$loglik,
    BIC              = best_model$BIC,
    AIC              = best_model$AIC,
    models           = models
  )
  
  class(res) <- "mvcwm_bin"
  return(res)
}
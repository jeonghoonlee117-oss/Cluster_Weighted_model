if (!requireNamespace("cluster", quietly = TRUE)) install.packages("cluster")
library(cluster)

# ==============================================================================
# Helper: optim(BFGS) 기반 가중 포아송 회귀계수 갱신 함수
# ==============================================================================
update_poisson_beta_optim <- function(X_design, y, alpha_i, beta_start) {
  
  # 1. 목적 함수: 음의 가중 로그 우도 (Negative Weighted Log-Likelihood)
  obj_fn <- function(beta) {
    eta <- as.vector(X_design %*% beta)
    eta <- pmin(pmax(eta, -700), 700) # 오버플로우/언더플로우 방지 클리핑
    lambda <- exp(eta)
    
    neg_loglik <- sum(alpha_i * (lambda - y * eta))
    return(neg_loglik)
  }
  
  # 2. 기울기 함수 (Gradient Vector)
  grad_fn <- function(beta) {
    eta <- as.vector(X_design %*% beta)
    eta <- pmin(pmax(eta, -700), 700)
    lambda <- exp(eta)
    
    grad <- crossprod(X_design, alpha_i * (lambda - y))
    return(as.vector(grad))
  }
  
  # 3. BFGS 최적화 실행
  optim_res <- optim(
    par     = beta_start,
    fn      = obj_fn,
    gr      = grad_fn,
    method  = "BFGS",
    control = list(maxit = 200, reltol = 1e-8)
  )
  
  return(optim_res$par)
}

# ==============================================================================
# 메인 함수: 다변량 포아송 군집 가중 모델 (mvcwm_poi)
# ==============================================================================
mvcwm_poi <- function(X, y, k = 1:5,
                      init_type = c("kmeans", "kmedoids", "kmedians"),
                      criterion = c("BIC", "AIC"),
                      max_iter = 1000, tol = 1e-6) {
  
  init_type <- match.arg(init_type)
  criterion <- match.arg(criterion)
  
  # 1. 입력 데이터 차원 정렬 및 유효성 검증
  X_mat <- as.matrix(X)
  N <- nrow(X_mat)
  D <- ncol(X_mat)
  y <- as.numeric(y)
  
  if (any(y < 0 | y != round(y))) {
    stop("포아송 모델의 반응 변수 y는 0 이상의 정수여야 합니다.")
  }
  
  # 설계 행렬 (Design Matrix)
  X_design <- cbind(1, X_mat)
  colnames(X_design) <- c("(Intercept)", if (!is.null(colnames(X))) colnames(X) else paste0("x", 1:D))
  
  # ----------------------------------------------------------------------------
  # 내부 단일 포아송 CWM 적합 함수
  # ----------------------------------------------------------------------------
  fit_single_mvcwm_poi <- function(M) {
    
    # (1) 초기 군집 분할
    if (M == 1) {
      init_cluster <- rep(1, N)
    } else {
      XY_mat <- cbind(scale(X_mat), scale(log1p(y)))
      init_cluster <- switch(
        init_type, 
        kmeans   = kmeans(XY_mat, centers = M, nstart = 20)$cluster,
        kmedoids = pam(XY_mat, k = M, metric = "euclidean")$clustering,
        kmedians = pam(XY_mat, k = M, metric = "manhattan")$clustering # [수정 7] manhanttan -> manhattan 오타 수정
      )
    }
    
    # (2) 초기 모수 설정
    w <- numeric(M)
    mu_x <- matrix(0, nrow = D, ncol = M)
    Sigma_x <- vector("list", M)
    Beta <- matrix(0, nrow = D + 1, ncol = M)
    
    # [수정 6] famliy -> family 오타 수정
    global_fit <- suppressWarnings(glm(y ~ X_mat, family = poisson(link = "log")))
    global_beta <- coef(global_fit)
    global_beta[is.na(global_beta)] <- 0
    
    for (i in 1:M) {
      idx <- which(init_cluster == i)
      n_i <- length(idx)
      w[i] <- n_i / N
      
      # X 분포 모수
      if (n_i <= 1) {
        mu_x[, i] <- if (n_i == 1) X_mat[idx, ] else colMeans(X_mat)
        Sigma_x[[i]] <- diag(1, D)
      } else {
        mu_x[, i] <- colMeans(X_mat[idx, , drop = FALSE])
        Sigma_x[[i]] <- cov(X_mat[idx, , drop = FALSE]) + diag(1e-6, D)
      }
      
      # [수정 1] 중복 else 문법 에러 수정
      if (n_i <= (D + 1)) {
        Beta[, i] <- global_beta
      } else {
        local_fit <- tryCatch(
          suppressWarnings(glm(y[idx] ~ X_mat[idx, , drop = FALSE], family = poisson(link = "log"))),
          error = function(e) NULL
        )
        if (!is.null(local_fit) && !any(is.na(coef(local_fit)))) {
          Beta[, i] <- coef(local_fit)
        } else {
          Beta[, i] <- global_beta
        }
      }
    }
    
    # (3) EM 알고리즘 루프
    loglik_history <- numeric()
    loglik_old <- -Inf
    
    for (iter in 1:max_iter) {
      
      # -------------------------
      # [E-Step]
      # -------------------------
      density_mat <- matrix(0, nrow = N, ncol = M)
      
      for (i in 1:M) {
        mu_i <- mu_x[, i]
        Sigma_i <- Sigma_x[[i]]
        mah_dist <- mahalanobis(X_mat, center = mu_i, cov = Sigma_i)
        
        # [수정 2] det() -> determinant() 정상 수정
        log_det <- as.numeric(determinant(Sigma_i, logarithm = TRUE)$modulus)
        log_p_x <- -0.5 * (D * log(2 * pi) + log_det + mah_dist)
        p_x <- exp(log_p_x)
        
        eta <- as.vector(X_design %*% Beta[, i])
        eta <- pmin(pmax(eta, -700), 700)
        lambda <- exp(eta)
        p_y_given_x <- dpois(y, lambda = pmax(lambda, 1e-10))
        
        density_mat[, i] <- w[i] * p_x * p_y_given_x
      }
      
      total_density <- pmax(rowSums(density_mat), 1e-300)
      alpha <- density_mat / total_density
      
      loglik_new <- sum(log(total_density))
      loglik_history <- c(loglik_history, loglik_new)
      
      if (abs(loglik_new - loglik_old) < tol) break
      loglik_old <- loglik_new
      
      # -------------------------
      # [M-Step]
      # -------------------------
      for (i in 1:M) {
        a <- alpha[, i]
        sum_a <- max(sum(a), 1e-12)
        
        w[i] <- sum_a / N
        
        mu_x[, i] <- colSums(a * X_mat) / sum_a
        X_centered <- sweep(X_mat, 2, mu_x[, i], "-")
        Sigma_x[[i]] <- (crossprod(X_centered, X_centered * a) / sum_a) + diag(1e-6, D)
        
        Beta[, i] <- update_poisson_beta_optim(
          X_design   = X_design,
          y          = y,
          alpha_i    = a, 
          beta_start = Beta[, i]
        )
      }
    }
    
    # 자유도 및 정보 기준 산출
    df <- (M - 1) + M * (2 * D + (D * (D + 1)) / 2 + 1)
    bic_val <- -2 * loglik_new + df * log(N)
    aic_val <- -2 * loglik_new + 2 * df # [수정 5] -2 + -> -2 * 오타 수정
    
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
      cluster = apply(alpha, 1, which.max) # [수정 4] alpah -> alpha 오타 수정
    ))
  }
  
  # ----------------------------------------------------------------------------
  # 후보 군집 모델 일괄 적합 및 최적 모델 선정
  # ----------------------------------------------------------------------------
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
    fit <- fit_single_mvcwm_poi(m_val)
    
    models[[idx]] <- fit
    # [수정 3] fix -> fit 오타 수정
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
  
  class(res) <- "mvcwm_poi"
  return(res)
}
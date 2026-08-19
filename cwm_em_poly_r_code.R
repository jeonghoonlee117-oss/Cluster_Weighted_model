if (!requireNamespace("cluster", quietly = TRUE)) install.packages("cluster")
library(cluster)

# ==============================================================================
# 메인 함수: 2차 다항 가우시안 군집 가중 모델 (cwm_gau_poly2)
# ==============================================================================
cwm_gau_poly2 <- function(x, y, k = 1:5,
                          init_type = c("kmeans", "kmedoids", "kmedians"),
                          criterion = c("BIC", "AIC"),
                          max_iter = 1000, tol = 1e-6) {
  
  init_type <- match.arg(init_type)
  criterion <- match.arg(criterion)
  
  x <- as.numeric(x)
  y <- as.numeric(y)
  N <- length(x)
  
  if (length(y) != N) {
    stop("x와 y의 길이가 일치해야 합니다.")
  }
  
  # ----------------------------------------------------------------------------
  # 내부 단일 모델 적합 함수
  # ----------------------------------------------------------------------------
  fit_single_cwm_poly2 <- function(M) {
    
    # (1) 초기 군집 분할
    if (M == 1) {
      init_cluster <- rep(1, N)
    } else {
      XY_mat <- cbind(scale(x), scale(y))
      init_cluster <- switch(
        init_type,
        kmeans   = kmeans(XY_mat, centers = M, nstart = 20)$cluster,
        kmedoids = pam(XY_mat, k = M, metric = "euclidean")$clustering,
        kmedians = pam(XY_mat, k = M, metric = "manhattan")$clustering
      )
    }
    
    # (2) 초기 모수 설정
    w <- numeric(M)
    mu_x <- numeric(M)
    sigma2_x <- numeric(M)
    Beta <- matrix(0, nrow = 3, ncol = M) # [beta0, beta1, beta2]'
    sigma2_y <- numeric(M)
    
    # 전체 2차 회귀 초기 적합 (안정적 초기값)
    global_fit <- lm(y ~ x + I(x^2))
    global_beta <- coef(global_fit)
    global_sigma2_y <- var(residuals(global_fit))
    
    for (m in 1:M) {
      idx <- which(init_cluster == m)
      n_m <- length(idx)
      w[m] <- n_m / N
      
      if (n_m <= 1) {
        mu_x[m] <- if (n_m == 1) x[idx] else mean(x)
        sigma2_x[m] <- var(x)
      } else {
        mu_x[m] <- mean(x[idx])
        sigma2_x[m] <- var(x[idx]) + 1e-6
      }
      
      if (n_m <= 3) {
        Beta[, m] <- global_beta
        sigma2_y[m] <- global_sigma2_y
      } else {
        local_fit <- tryCatch(lm(y[idx] ~ x[idx] + I(x[idx]^2)), error = function(e) NULL)
        if (!is.null(local_fit) && !any(is.na(coef(local_fit)))) {
          Beta[, m] <- coef(local_fit)
          sigma2_y[m] <- var(residuals(local_fit)) + 1e-6
        } else {
          Beta[, m] <- global_beta
          sigma2_y[m] <- global_sigma2_y
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
      
      for (m in 1:M) {
        # 1) 입력 확률밀도 p(x)
        p_x <- dnorm(x, mean = mu_x[m], sd = sqrt(sigma2_x[m]))
        
        # 2) 조건부 출력 확률밀도 p(y | x) = N(beta0 + beta1*x + beta2*x^2, sigma2_y)
        y_hat <- Beta[1, m] + Beta[2, m] * x + Beta[3, m] * (x^2)
        p_y_given_x <- dnorm(y, mean = y_hat, sd = sqrt(sigma2_y[m]))
        
        density_mat[, m] <- w[m] * p_x * p_y_given_x
      }
      
      total_density <- pmax(rowSums(density_mat), 1e-300)
      alpha <- density_mat / total_density
      
      loglik_new <- sum(log(total_density))
      loglik_history <- c(loglik_history, loglik_new)
      
      if (abs(loglik_new - loglik_old) < tol) break
      loglik_old <- loglik_new
      
      # -------------------------
      # [M-Step] : 제시해주신 S, T 행렬 공식 적용
      # -------------------------
      for (m in 1:M) {
        a <- alpha[, m]
        sum_a <- max(sum(a), 1e-12)
        
        # 1) 혼합 비율 w_m
        w[m] <- sum_a / N
        
        # 2) X 분포 모수 갱신
        mu_x[m] <- sum(a * x) / sum_a
        sigma2_x[m] <- (sum(a * (x - mu_x[m])^2) / sum_a) + 1e-6
        
        # 3) S 행렬 (Hankel 형태) 계산
        S0 <- sum_a
        S1 <- sum(a * x)
        S2 <- sum(a * (x^2))
        S3 <- sum(a * (x^3))
        S4 <- sum(a * (x^4))
        
        S_mat <- matrix(c(S0, S1, S2,
                          S1, S2, S3,
                          S2, S3, S4), nrow = 3, byrow = TRUE)
        
        # 4) T 벡터 계산
        T0 <- sum(a * y)
        T1 <- sum(a * x * y)
        T2 <- sum(a * (x^2) * y)
        
        T_vec <- c(T0, T1, T2)
        
        # 5) 2차 회귀계수 갱신: Beta = S^{-1} * T
        Beta[, m] <- solve(S_mat + diag(1e-8, 3), T_vec)
        
        # 6) 잔차 분산 갱신
        y_fitted <- Beta[1, m] + Beta[2, m] * x + Beta[3, m] * (x^2)
        sigma2_y[m] <- (sum(a * (y - y_fitted)^2) / sum_a) + 1e-6
      }
    }
    
    # 자유도 계산 (모수: w: M-1, mu_x: M, sigma2_x: M, Beta: 3M, sigma2_y: M -> 총 7M - 1)
    df <- (M - 1) + M * (1 + 1 + 3 + 1)
    bic_val <- -2 * loglik_new + df * log(N)
    aic_val <- -2 * loglik_new + 2 * df
    
    rownames(Beta) <- c("beta0 (Intercept)", "beta1 (x)", "beta2 (x^2)")
    
    return(list(
      M = M, N = N, df = df,
      iterations = iter,
      loglik_history = loglik_history,
      loglik = loglik_new,
      BIC = bic_val,
      AIC = aic_val,
      w = w, mu_x = mu_x, sigma2_x = sigma2_x,
      Beta = Beta, sigma2_y = sigma2_y,
      alpha = alpha,
      cluster = apply(alpha, 1, which.max)
    ))
  }
  
  # ----------------------------------------------------------------------------
  # 후보 군집 일괄 탐색 및 최적 모델 선택
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
    fit <- fit_single_cwm_poly2(m_val)
    
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
    sigma2_x         = best_model$sigma2_x,
    Beta             = best_model$Beta,
    sigma2_y         = best_model$sigma2_y,
    loglik           = best_model$loglik,
    BIC              = best_model$BIC,
    AIC              = best_model$AIC,
    models           = models
  )
  
  class(res) <- "cwm_gau_poly2"
  return(res)
}
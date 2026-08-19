if (!requireNamespace("cluster", quietly = TRUE)) install.packages("cluster")
library(cluster)

cwm <- function(X, y, k = 1:5,
                init_type = c("kmeans", "kmedoids", "kmedians"),
                criterion = c("BIC", "AIC"),
                max_iter = 1000, tol = 1e-10) {
  
  init_type <- match.arg(init_type)
  criterion <- match.arg(criterion)
  
  # 다변량 분석을 위한 매트릭스 지정
  X_mat <- as.matrix(X)
  N <- nrow(X_mat)
  D <- ncol(X_mat)
  y <- as.numeric(y)
  
  # 다변량 절편을 위한 모델 처리 (Design Matrix)
  X_design <- cbind(1, X_mat)
  colnames(X_design) <- c("(Intercept)", if (!is.null(colnames(X))) colnames(X) else paste0("x", 1:D))
  
  # ----------------------------------------------------------------------------
  # 내부 단일 CWM 적합 함수
  # ----------------------------------------------------------------------------
  fit_single_cwm <- function(M) {
    
    # 1. 초기 군집 할당
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
    
    # 2. 초기 모수 세팅
    w <- numeric(M)
    mu_x <- matrix(0, nrow = D, ncol = M)
    Sigma_x <- vector("list", M)
    Beta <- matrix(0, nrow = D + 1, ncol = M)
    sigma_y <- numeric(M)
    
    for (i in 1:M) {
      idx <- which(init_cluster == i)
      n_i <- length(idx)
      w[i] <- n_i / N
      
      # [수정 2] 표본 수가 부족할 때 NA 방지 예외 처리
      if (n_i <= 1) {
        mu_x[, i] <- if (n_i == 1) X_mat[idx, ] else colMeans(X_mat)
        Sigma_x[[i]] <- diag(1, D)
      } else {
        mu_x[, i] <- colMeans(X_mat[idx, , drop = FALSE])
        Sigma_x[[i]] <- cov(X_mat[idx, , drop = FALSE]) + diag(1e-6, D)
      }
      
      if (n_i <= (D + 1)) {
        fit_lm <- lm(y ~ X_mat)
      } else {
        fit_lm <- lm(y[idx] ~ X_mat[idx, , drop = FALSE])
      }
      
      Beta[, i] <- coef(fit_lm)
      sigma_y[i] <- max(sd(residuals(fit_lm)), 1e-6)
    }
    
    # 3. EM 알고리즘 루프
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
        
        y_pred <- as.vector(X_design %*% Beta[, i])
        p_y_given_x <- dnorm(y, mean = y_pred, sd = sigma_y[i])
        
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
        
        Xt_W_X <- crossprod(X_design, X_design * a)
        Xt_W_y <- crossprod(X_design, y * a)
        Beta[, i] <- solve(Xt_W_X + diag(1e-8, D + 1), Xt_W_y)
        
        residuals_new <- y - as.vector(X_design %*% Beta[, i])
        sigma_y[i] <- max(sqrt(sum(a * (residuals_new^2)) / sum_a), 1e-6)
      }
    }
    
    # 정보 기준 산출
    df <- (M - 1) + M * (2 * D + (D * (D + 1)) / 2 + 2)
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
      Beta = Beta, sigma_y = sigma_y,
      alpha = alpha,
      cluster = apply(alpha, 1, which.max)
    ))
  }
  
  # ----------------------------------------------------------------------------
  # 후보 군집 모델 일괄 평가
  # ----------------------------------------------------------------------------
  k_candidates <- sort(unique(as.integer(k)))
  num_k <- length(k_candidates)
  
  models <- vector("list", num_k)
  names(models) <- paste0("M=", k_candidates)
  
  comparison_table <- data.frame(
    M = k_candidates,                      # [수정 3] 쉼표(,) 추가
    LogLik = numeric(num_k),
    df = numeric(num_k),
    BIC = numeric(num_k),
    AIC = numeric(num_k)
  )
  
  for (idx in seq_along(k_candidates)) {
    m_val <- k_candidates[idx]
    fit <- fit_single_cwm(m_val)
    
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
    best_M = best_model$M,
    criterion = criterion,
    comparison_table = comparison_table,
    cluster = best_model$cluster,
    posterior = best_model$alpha,
    w = best_model$w, 
    mu_x = best_model$mu_x,
    Sigma_x = best_model$Sigma_x,
    Beta = best_model$Beta,
    sigma_y = best_model$sigma_y,
    loglik = best_model$loglik,
    BIC = best_model$BIC,
    AIC = best_model$AIC,
    models = models
  )
  
  class(res) <- "cwm"
  return(res)
}

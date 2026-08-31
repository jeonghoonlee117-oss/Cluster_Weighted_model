"""
GCWM (Gaussian linear Cluster-Weighted Model) - EM algorithm
논문 Eq. 2, 3, 4 및 EM 업데이트 식을 그대로 구현.

p(x, y) = sum_k pi_k * phi_p(x; mu_k, Sigma_k) * phi_1(y; (1,x)beta_k, sigma_k^2)
"""
import numpy as np
from scipy.stats import multivariate_normal, norm


def _design_matrix(X):
    """X: (N,p) -> A: (N,p+1) with leading column of 1s (intercept)."""
    N = X.shape[0]
    return np.hstack([np.ones((N, 1)), X])


def gcwm_em(X, y, K, n_init=10, max_iter=500, tol=1e-5, random_state=None,
            use_d_deleted=True, d=None):
    """
    Fit a linear Gaussian CWM with K components via EM, with multiple
    random initializations (k-means-like init on 90% subsample as in
    the paper's Section 4).

    Model selection among the n_init candidate solutions follows the
    paper's Section 4: rather than picking the run with the largest
    ordinary log-likelihood (which is prone to singular/spurious
    solutions), we pick the run with the largest gradient-based
    d-deleted log-likelihood (Kim & Seo, 2014) when use_d_deleted=True.

    Parameters
    ----------
    X : (N, p) array of covariates
    y : (N,) array of responses
    K : number of components
    n_init : number of random initializations to try
    use_d_deleted : if True, select best init by d-deleted log-likelihood;
                    if False, fall back to plain log-likelihood (old behavior)
    d : number of most-influential observations to delete (default p+1)

    Returns
    -------
    dict with fitted parameters, responsibilities, log-likelihood, BIC, ICL,
    plus (if use_d_deleted) the d-deleted diagnostics under key 'd_deleted'.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    N, p = X.shape
    A = _design_matrix(X)

    candidates = []
    for init in range(n_init):
        try:
            fit = _gcwm_em_single(X, y, A, K, max_iter, tol, rng)
        except np.linalg.LinAlgError:
            continue
        if fit is None:
            continue
        candidates.append(fit)
    if not candidates:
        raise RuntimeError("All EM initializations failed (likely singular).")

    if use_d_deleted:
        for fit in candidates:
            fit["d_deleted"] = d_deleted_loglik(X, y, fit, d=d)
        best = max(candidates, key=lambda f: f["d_deleted"]["ell_minus_d"])
    else:
        best = max(candidates, key=lambda f: f["loglik"])

    # Number of free parameters, following paper's formula for Eq.2:
    # K(p^2+3p+2)/2 - 1 + K(p+2)
    n_params = K * (p**2 + 3 * p + 2) / 2 - 1 + K * (p + 2)
    best["n_params"] = n_params
    best["bic"] = -2 * best["loglik"] + n_params * np.log(N)
    best["icl"] = best["bic"] - 2 * _entropy_term(best["tau"])
    best["K"] = K
    best["n_candidates"] = len(candidates)
    return best


def _entropy_term(tau):
    """Sum of entropies used in ICL = BIC - 2*sum_i sum_k tau_ik*log(tau_ik)."""
    t = np.clip(tau, 1e-300, 1.0)
    return np.sum(t * np.log(t))


def _per_obs_log_density(X, y, A, pi, mu, Sigma, beta, sigma2, K):
    """Return (N,K) matrix of component densities phi_j(x_i,y_i) (NOT log),
    and (N,) vector of log p(x_i,y_i) = log sum_k pi_k * phi_k."""
    N = X.shape[0]
    log_phi = np.zeros((N, K))  # log phi_j(x_i,y_i), WITHOUT pi_j
    for k in range(K):
        log_phi_x = multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k])
        mean_y = A @ beta[k]
        log_phi_y = norm.logpdf(y, loc=mean_y, scale=np.sqrt(sigma2[k]))
        log_phi[:, k] = log_phi_x + log_phi_y

    log_pi = np.log(pi + 1e-300)
    log_mix = log_pi[None, :] + log_phi  # (N,K), log(pi_k * phi_k)
    max_log = np.max(log_mix, axis=1, keepdims=True)
    log_p_xy = (max_log.ravel() +
                np.log(np.sum(np.exp(log_mix - max_log), axis=1)))
    return log_phi, log_pi, log_p_xy


def d_deleted_loglik(X, y, fit, d=None):
    """
    Compute the gradient-based d-deleted log-likelihood (Kim & Seo, 2014)
    as described in the paper's Section 4, to guard against singular /
    spurious EM solutions.

    g_{i,j} = phi_j(x_i,y_i) / sum_k pi_k*phi_k(x_i,y_i) - 1,  j=1,...,K-1
    I_i     = g_i' (G'G)^{-1} g_i         (influence of observation i)
    R       = indices of the d largest I_i
    ell_{-d}(theta) = ell(theta) - sum_{r in R} log p(x_r, y_r)

    Returns the observations' influence I_i as well, for diagnostics.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    A = _design_matrix(X)
    N, p = X.shape
    K = fit["pi"].shape[0]
    if d is None:
        d = p + 1

    log_phi, log_pi, log_p_xy = _per_obs_log_density(
        X, y, A, fit["pi"], fit["mu"], fit["Sigma"], fit["beta"], fit["sigma2"], K)

    # phi_j(x_i,y_i) / sum_k pi_k phi_k(x_i,y_i) for j=1..K-1
    # = exp( log_phi[:,j] - log_p_xy )
    ratio = np.exp(log_phi[:, :K - 1] - log_p_xy[:, None])  # (N, K-1)
    G = ratio - 1.0  # g_{i,j}, shape (N, K-1)

    # Regularize (G'G) in case K-1 > 0 but near-singular
    GtG = G.T @ G
    GtG += 1e-8 * np.eye(K - 1) if K > 1 else np.eye(0)

    if K == 1:
        # No g_{i,j} defined (K-1=0): influence is trivially zero.
        I = np.zeros(N)
    else:
        GtG_inv = np.linalg.pinv(GtG)
        I = np.einsum('ij,jk,ik->i', G, GtG_inv, G)

    ell = fit["loglik"]
    order = np.argsort(-I)  # descending influence
    R = order[:d]
    ell_minus_d = ell - np.sum(log_p_xy[R])

    return dict(I=I, R=R, ell=ell, ell_minus_d=ell_minus_d, d=d)


def _kmeans_init(X, y, K, rng, subsample_frac=0.9):
    """Rough k-means-style init: subsample 90%, run simple k-means, then
    fit each component's (mu,Sigma,beta,sigma2) on its subsample chunk."""
    N = X.shape[0]
    n_sub = max(K + 2, int(subsample_frac * N))
    idx = rng.choice(N, size=n_sub, replace=False)
    Xs, ys = X[idx], y[idx]

    # simple k-means (few iterations) on Xs
    centers = Xs[rng.choice(n_sub, size=K, replace=False)]
    for _ in range(20):
        d = np.linalg.norm(Xs[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        new_centers = np.array([
            Xs[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
            for k in range(K)
        ])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    p = X.shape[1]
    mu = np.zeros((K, p))
    Sigma = np.zeros((K, p, p))
    beta = np.zeros((K, p + 1))
    sigma2 = np.zeros(K)
    pi = np.zeros(K)

    for k in range(K):
        mask = labels == k
        if mask.sum() < p + 2:
            # fallback: random small subset
            mask = rng.choice(n_sub, size=p + 2, replace=False)
            mask_bool = np.zeros(n_sub, dtype=bool)
            mask_bool[mask] = True
            mask = mask_bool
        Xk, yk = Xs[mask], ys[mask]
        mu[k] = Xk.mean(axis=0)
        cov = np.cov(Xk.T) if p > 1 else np.array([[Xk.var() + 1e-6]])
        Sigma[k] = np.atleast_2d(cov) + 1e-6 * np.eye(p)
        Ak = _design_matrix(Xk)
        beta_k, *_ = np.linalg.lstsq(Ak, yk, rcond=None)
        beta[k] = beta_k
        resid = yk - Ak @ beta_k
        sigma2[k] = max(np.mean(resid**2), 1e-4)
        pi[k] = mask.sum() / n_sub

    pi = pi / pi.sum()
    return pi, mu, Sigma, beta, sigma2


def _gcwm_em_single(X, y, A, K, max_iter, tol, rng):
    N, p = X.shape
    pi, mu, Sigma, beta, sigma2 = _kmeans_init(X, y, K, rng)

    prev_ll = -np.inf
    for it in range(max_iter):
        # ---- E-step ----
        log_dens = np.zeros((N, K))
        for k in range(K):
            try:
                log_phi_x = multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k])
            except np.linalg.LinAlgError:
                return None
            mean_y = A @ beta[k]
            log_phi_y = norm.logpdf(y, loc=mean_y, scale=np.sqrt(sigma2[k]))
            log_dens[:, k] = np.log(pi[k] + 1e-300) + log_phi_x + log_phi_y

        max_log = np.max(log_dens, axis=1, keepdims=True)
        w = np.exp(log_dens - max_log)
        row_sum = w.sum(axis=1, keepdims=True)
        tau = w / row_sum
        loglik = np.sum(max_log.ravel() + np.log(row_sum.ravel()))

        if not np.isfinite(loglik):
            return None

        # ---- M-step ----
        Nk = tau.sum(axis=0)
        if np.any(Nk < p + 1):  # degenerate component
            return None
        pi = Nk / N
        for k in range(K):
            w_k = tau[:, k]
            mu[k] = (w_k[:, None] * X).sum(axis=0) / Nk[k]
            Xc = X - mu[k]
            Sigma[k] = (Xc * w_k[:, None]).T @ Xc / Nk[k] + 1e-6 * np.eye(p)

            Wk = w_k
            AtWA = A.T @ (Wk[:, None] * A)
            AtWy = A.T @ (Wk * y)
            try:
                beta[k] = np.linalg.solve(AtWA, AtWy)
            except np.linalg.LinAlgError:
                return None
            resid = y - A @ beta[k]
            sigma2[k] = max((Wk * resid**2).sum() / Nk[k], 1e-6)

        if abs(loglik - prev_ll) < tol * abs(prev_ll) + 1e-10:
            prev_ll = loglik
            break
        prev_ll = loglik

    return dict(pi=pi, mu=mu, Sigma=Sigma, beta=beta, sigma2=sigma2,
                tau=tau, loglik=prev_ll, n_iter=it + 1)

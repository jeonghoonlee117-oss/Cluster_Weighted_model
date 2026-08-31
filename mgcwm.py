"""
MGCWM: Merged Gaussian linear Cluster-Weighted Model
논문 Algorithm 1을 구현.
1) GCWM(K components)을 적합
2) beta 사이 L1 거리가 가장 가까운 두 클러스터를 병합 후보로 삼음
3) 병합한 reduced model(beta 공유)을 EM으로 재적합, BIC/ICL이 개선되면 확정
4) 더 이상 개선되는 병합이 없을 때까지 반복
"""
import numpy as np
from scipy.stats import multivariate_normal, norm
from itertools import combinations
from gcwm import _design_matrix, _entropy_term


def _refit_partition(X, y, gcwm_fit, groups, max_iter=500, tol=1e-5):
    """
    Given the original K-component GCWM fit and a partition `groups`
    (list of lists of component indices, e.g. [[0],[1,4],[2],[3]]),
    re-estimate all parameters of the merged model via EM:
      - mu_k, Sigma_k, sigma_k^2 remain per-component (k)
      - beta_m is SHARED within each cluster (m)
    Initial values: taken from gcwm_fit, with beta_m initialized as the
    average of the beta_k's in that cluster (as stated in the paper).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    N, p = X.shape
    A = _design_matrix(X)
    K = gcwm_fit["pi"].shape[0]
    M = len(groups)

    # map component -> cluster
    comp2cluster = np.zeros(K, dtype=int)
    for m, g in enumerate(groups):
        for k in g:
            comp2cluster[k] = m

    mu = gcwm_fit["mu"].copy()
    Sigma = gcwm_fit["Sigma"].copy()
    sigma2 = gcwm_fit["sigma2"].copy()
    eta = gcwm_fit["pi"].copy()  # eta_k = p(w=m,z=k) ~ initialize with pi_k
    beta_cluster = np.zeros((M, p + 1))
    for m, g in enumerate(groups):
        beta_cluster[m] = gcwm_fit["beta"][g].mean(axis=0)

    prev_ll = -np.inf
    tau = None
    for it in range(max_iter):
        # ---- E-step: responsibility for each of the K components ----
        log_dens = np.zeros((N, K))
        for k in range(K):
            m = comp2cluster[k]
            try:
                log_phi_x = multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k])
            except np.linalg.LinAlgError:
                return None
            mean_y = A @ beta_cluster[m]
            log_phi_y = norm.logpdf(y, loc=mean_y, scale=np.sqrt(sigma2[k]))
            log_dens[:, k] = np.log(eta[k] + 1e-300) + log_phi_x + log_phi_y

        max_log = np.max(log_dens, axis=1, keepdims=True)
        w = np.exp(log_dens - max_log)
        row_sum = w.sum(axis=1, keepdims=True)
        tau = w / row_sum  # (N,K), tau[:,k] = w_imk in the paper's notation
        loglik = np.sum(max_log.ravel() + np.log(row_sum.ravel()))
        if not np.isfinite(loglik):
            return None

        # ---- M-step ----
        Nk = tau.sum(axis=0)
        if np.any(Nk < p + 1):
            return None
        eta = Nk / N
        for k in range(K):
            w_k = tau[:, k]
            mu[k] = (w_k[:, None] * X).sum(axis=0) / Nk[k]
            Xc = X - mu[k]
            Sigma[k] = (Xc * w_k[:, None]).T @ Xc / Nk[k] + 1e-6 * np.eye(p)

        # beta_m: pooled weighted least squares over all components in cluster m
        for m, g in enumerate(groups):
            w_m = tau[:, g].sum(axis=1)  # w_im. = sum_{k in Gm} w_imk
            AtWA = A.T @ (w_m[:, None] * A)
            AtWy = A.T @ (w_m * y)
            try:
                beta_cluster[m] = np.linalg.solve(AtWA, AtWy)
            except np.linalg.LinAlgError:
                return None

        for k in range(K):
            m = comp2cluster[k]
            w_k = tau[:, k]
            resid = y - A @ beta_cluster[m]
            sigma2[k] = max((w_k * resid**2).sum() / Nk[k], 1e-6)

        if abs(loglik - prev_ll) < tol * abs(prev_ll) + 1e-10:
            prev_ll = loglik
            break
        prev_ll = loglik

    n_params = K * (p**2 + 3 * p + 2) / 2 - 1 + M * (p + 1) + K
    bic = -2 * prev_ll + n_params * np.log(N)
    icl = bic - 2 * _entropy_term(tau)

    return dict(mu=mu, Sigma=Sigma, sigma2=sigma2, eta=eta,
                beta_cluster=beta_cluster, groups=groups,
                comp2cluster=comp2cluster, tau=tau,
                loglik=prev_ll, bic=bic, icl=icl,
                n_params=n_params, M=M)


def merge_gcwm(X, y, gcwm_fit, criterion="bic", verbose=True):
    """
    Algorithm 1 from the paper.

    Parameters
    ----------
    gcwm_fit : output of gcwm_em() -- the K-component GCWM fit to start from
    criterion: "bic" or "icl"

    Returns
    -------
    list of dicts, one per step of the algorithm (the history of accepted
    merges), the last one being the final MGCWM model.
    """
    K = gcwm_fit["pi"].shape[0]
    groups = [[k] for k in range(K)]  # start: M=K, each component alone

    current = _refit_partition(X, y, gcwm_fit, groups)
    if current is None:
        raise RuntimeError("Initial refit (M=K) failed.")
    history = [current]

    if verbose:
        print(f"Start: M={len(groups)}, {criterion.upper()}={current[criterion]:.3f}")

    while len(groups) > 1:
        # candidate pairs sorted by L1 distance between cluster betas
        beta_now = current["beta_cluster"]
        dists = []
        for (i, j) in combinations(range(len(groups)), 2):
            d = np.sum(np.abs(beta_now[i] - beta_now[j]))
            dists.append((d, i, j))
        dists.sort(key=lambda t: t[0])

        accepted = False
        for d, i, j in dists:
            trial_groups = [g for idx, g in enumerate(groups) if idx not in (i, j)]
            trial_groups.append(groups[i] + groups[j])

            trial_fit = _refit_partition(X, y, gcwm_fit, trial_groups)
            if trial_fit is None:
                continue

            if trial_fit[criterion] < current[criterion]:
                groups = trial_groups
                current = trial_fit
                history.append(current)
                accepted = True
                if verbose:
                    print(f"Merge accepted (d={d:.3f}): M={len(groups)}, "
                          f"{criterion.upper()}={current[criterion]:.3f}")
                break  # restart search from the new partition
        if not accepted:
            if verbose:
                print(f"No further improving merge found. Final M={len(groups)}.")
            break

    return history

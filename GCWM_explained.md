# GCWM (Gaussian Linear Cluster-Weighted Model) — 코드와 수식 대조

이 문서는 `gcwm.py`의 각 함수가 논문(Oh & Seo, 2022) 수식의 어느 부분을 구현한 것인지 정리한다.

---

## 1. 모형 정의

GCWM은 결합확률밀도 $p(x,y)$를 다음과 같이 모델링한다.

$$
p(x, y) = \sum_{k=1}^{K} \pi_k \, \phi_p(x;\mu_k,\Sigma_k)\, \phi_1\big(y;(1,x)\beta_k,\ \sigma_k^2\big)
$$

- $\pi_k$ : 컴포넌트 $k$의 혼합비율, $\sum_k \pi_k = 1$
- $\phi_p(x;\mu_k,\Sigma_k)$ : $p$차원 공변량 $x$의 컴포넌트별 정규분포
- $\phi_1\big(y;(1,x)\beta_k,\sigma_k^2\big)$ : $x$가 주어졌을 때 $y$의 조건부 정규분포 (선형회귀 평균, 등분산)

---

## 2. `_design_matrix(X)` — 설계행렬 구성

$$
A =
\begin{pmatrix}
1 & x_1^\top \\
1 & x_2^\top \\
\vdots & \vdots \\
1 & x_N^\top
\end{pmatrix}
\in \mathbb{R}^{N\times(p+1)}
$$

회귀 평균 $(1,x_i)\beta_k$를 행렬 곱 $A\beta_k$ 하나로 계산하기 위한 준비 단계이다.

```python
def _design_matrix(X):
    N = X.shape[0]
    return np.hstack([np.ones((N, 1)), X])
```

---

## 3. `_gcwm_em_single` — EM 알고리즘 한 번의 실행

### 3.1 E-step

현재 파라미터 $\theta^{(t)}$ 하에서, 관측치 $i$가 컴포넌트 $k$에서 나왔을 사후확률(=$w_{ik}^{(t)}$, 코드에서는 `tau`)을 계산한다.

$$
w_{ik}^{(t)} = \frac{\pi_k^{(t)}\,\phi_p\!\big(x_i;\mu_k^{(t)},\Sigma_k^{(t)}\big)\,\phi_1\!\big(y_i;(1,x_i)\beta_k^{(t)},\sigma_k^{2(t)}\big)}
{\displaystyle\sum_{l=1}^{K} \pi_l^{(t)}\,\phi_p\!\big(x_i;\mu_l^{(t)},\Sigma_l^{(t)}\big)\,\phi_1\!\big(y_i;(1,x_i)\beta_l^{(t)},\sigma_l^{2(t)}\big)}
$$

수치안정성을 위해 로그 스케일(log-sum-exp)로 계산한다.

```python
log_dens[:, k] = np.log(pi[k] + 1e-300) + log_phi_x + log_phi_y
max_log = np.max(log_dens, axis=1, keepdims=True)
w = np.exp(log_dens - max_log)
tau = w / w.sum(axis=1, keepdims=True)
```

관측 로그우도는

$$
\ell(\theta^{(t)}) = \sum_{i=1}^{N} \log\!\left(\sum_{k=1}^{K}\pi_k^{(t)}\,\phi_p\phi_1\right)
$$

로그-합-지수(log-sum-exp) 트릭을 이용해

$$
\ell(\theta^{(t)}) = \sum_{i=1}^N \left[\max_k(\text{log\_dens}_{ik}) + \log\!\sum_{k} \exp\big(\text{log\_dens}_{ik}-\max_k(\text{log\_dens}_{ik})\big)\right]
$$

로 계산된다.

---

### 3.2 M-step

$Q$함수를 각 파라미터로 미분해 얻어지는 닫힌 형태(closed-form) 업데이트 식이다.

**혼합비율**

$$
\pi_k^{(t+1)} = \frac{N_k}{N}, \qquad N_k \equiv \sum_{i=1}^N w_{ik}^{(t)}
$$

```python
Nk = tau.sum(axis=0)
pi = Nk / N
```

**평균 (가중평균)**

$$
\mu_k^{(t+1)} = \frac{\sum_{i=1}^N w_{ik}^{(t)}\,x_i}{\sum_{i=1}^N w_{ik}^{(t)}}
$$

```python
mu[k] = (w_k[:, None] * X).sum(axis=0) / Nk[k]
```

**공분산 (가중 외적)**

$$
\Sigma_k^{(t+1)} = \frac{\sum_{i=1}^N w_{ik}^{(t)}\,(x_i-\mu_k^{(t+1)})(x_i-\mu_k^{(t+1)})^\top}{\sum_{i=1}^N w_{ik}^{(t)}}
$$

```python
Xc = X - mu[k]
Sigma[k] = (Xc * w_k[:, None]).T @ Xc / Nk[k] + 1e-6 * np.eye(p)
```

($10^{-6}I_p$는 특이행렬 방지를 위한 릿지 보정, 논문에는 없는 구현상의 안전장치)

**회귀계수 (가중최소제곱, WLS)**

$$
\beta_k^{(t+1)} = \big(A^\top W_k^{(t)} A\big)^{-1} A^\top W_k^{(t)} y,
\qquad W_k^{(t)} = \mathrm{diag}\big(w_{1k}^{(t)},\dots,w_{Nk}^{(t)}\big)
$$

```python
AtWA = A.T @ (Wk[:, None] * A)
AtWy = A.T @ (Wk * y)
beta[k] = np.linalg.solve(AtWA, AtWy)
```

**오차분산**

$$
\sigma_k^{2(t+1)} = \frac{\sum_{i=1}^N w_{ik}^{(t)}\,\big(y_i-(1,x_i)\beta_k^{(t+1)}\big)^2}{\sum_{i=1}^N w_{ik}^{(t)}}
$$

```python
resid = y - A @ beta[k]
sigma2[k] = max((Wk * resid**2).sum() / Nk[k], 1e-6)
```

---

### 3.3 수렴 판정

$$
\left|\frac{\ell(\theta^{(t+1)}) - \ell(\theta^{(t)})}{\ell(\theta^{(t)})}\right| < 10^{-5}
\quad \text{또는} \quad t \ge 500
$$

```python
if abs(loglik - prev_ll) < tol * abs(prev_ll) + 1e-10:
    break
```

---

## 4. `_kmeans_init` — 초기값 생성

논문 Section 4의 절차를 그대로 구현한다.

1. 이상치 영향을 줄이기 위해 데이터의 90%를 무작위 서브샘플링

$$
n_{\text{sub}} = \max(K+2,\ 0.9N)
$$

2. 그 서브샘플에 대해 k-means로 $K$개 그룹을 만듦
3. 각 그룹 $k$ 안에서 독립적으로

$$
\hat\mu_k^{(0)} = \bar{x}_k,\qquad
\hat\Sigma_k^{(0)} = \widehat{\mathrm{Cov}}(x_k),\qquad
\hat\beta_k^{(0)} = (A_k^\top A_k)^{-1}A_k^\top y_k,\qquad
\hat\sigma_k^{2(0)} = \frac{1}{n_k}\sum(y_i-\hat{y}_i)^2
$$

를 계산해 초기값으로 삼는다. 이 전체 과정을 `n_init`회(기본 10회) 반복하여 서로 다른 후보 해를 얻는다.

---

## 5. `d_deleted_loglik` — Singular / Spurious solution 방어

### 5.1 배경

혼합우도는 이론적으로 비유계(unbounded)이므로, 단순히 "로그우도가 가장 큰 해"를 고르면 소수 관측치에 병적으로 과적합된 해(singular / spurious solution)를 선택할 위험이 있다. 이를 막기 위해 Kim & Seo (2014)의 gradient-based $d$-deleted log-likelihood를 사용한다.

### 5.2 관측치별 스코어

$$
g_{i,j} = \frac{\phi_p(x_i;\hat\mu_j,\hat\Sigma_j)\,\phi_1(y_i;(1,x_i)\hat\beta_j,\hat\sigma_j^2)}
{\sum_{k=1}^{K}\hat\pi_k\,\phi_p(x_i;\hat\mu_k,\hat\Sigma_k)\,\phi_1(y_i;(1,x_i)\hat\beta_k,\hat\sigma_k^2)} - 1,
\qquad j=1,\dots,K-1
$$

```python
ratio = np.exp(log_phi[:, :K - 1] - log_p_xy[:, None])
G = ratio - 1.0   # (N, K-1), 이게 g_{i,j}
```

### 5.3 영향력 (마할라노비스형 이차형식)

$$
g_i = (g_{i,1},\dots,g_{i,K-1})^\top,\qquad
G = (g_1,\dots,g_N)^\top,\qquad
I_i = g_i^\top (G^\top G)^{-1} g_i
$$

이는 마할라노비스 거리 $D_i^2=(x_i-\bar x)^\top S^{-1}(x_i-\bar x)$와 동일한 구조로, "이 관측치가 특정 컴포넌트에 얼마나 이례적으로(disproportionately) 잘 맞는가"를 측정한다.

```python
GtG = G.T @ G + 1e-8 * np.eye(K - 1)
GtG_inv = np.linalg.pinv(GtG)
I = np.einsum('ij,jk,ik->i', G, GtG_inv, G)
```

### 5.4 d-deleted log-likelihood

영향력이 가장 큰 $d$개 관측치 집합 $R$을 구하고, 그 기여분을 우도에서 제거한다.

$$
\ell_{-d}(\theta) = \ell(\theta) - \sum_{r\in R}\log p(x_r,y_r), \qquad d = p+1 \ \text{(기본값)}
$$

```python
order = np.argsort(-I)
R = order[:d]
ell_minus_d = ell - np.sum(log_p_xy[R])
```

`gcwm_em`은 `n_init`개의 후보 중 **$\ell(\theta)$가 아니라 $\ell_{-d}(\theta)$가 가장 큰 해**를 최종 선택한다.

$$
\hat\theta = \operatorname*{argmax}_{\theta \in \{\theta^{(1)},\dots,\theta^{(n_{\text{init}})}\}} \ell_{-d}(\theta)
$$

---

## 6. 모형 선택 기준 — BIC / ICL

$$
n_{\text{params}} = K\cdot\frac{p^2+3p+2}{2} - 1 + K(p+2)
$$

$$
\mathrm{BIC} = -2\hat\ell + n_{\text{params}}\log N
$$

$$
\mathrm{ICL} = \mathrm{BIC} - 2\sum_{i=1}^N\sum_{k=1}^K w_{ik}\log w_{ik}
$$

```python
n_params = K * (p**2 + 3 * p + 2) / 2 - 1 + K * (p + 2)
bic = -2 * best["loglik"] + n_params * np.log(N)
icl = bic - 2 * _entropy_term(best["tau"])
```

---

## 7. 전체 흐름 요약

$$
\underbrace{\text{90\% 서브샘플 k-means}}_{\S 4}
\;\xrightarrow{n_{\text{init}}\text{회 반복}}\;
\underbrace{\text{E-step}\leftrightarrow\text{M-step}}_{\S 3}
\;\xrightarrow{\text{수렴}}\;
\underbrace{d\text{-deleted log-likelihood 계산}}_{\S 5}
\;\xrightarrow{\text{최댓값 선택}}\;
\underbrace{\mathrm{BIC}/\mathrm{ICL}}_{\S 6}
$$

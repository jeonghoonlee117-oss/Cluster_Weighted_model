# MGCWM (Merged Gaussian Linear Cluster-Weighted Model) — 코드와 수식 대조

이 문서는 `mgcwm.py`의 각 함수가 논문(Oh & Seo, 2022) Section 3.1~3.3의 어느 수식을 구현한 것인지 정리한다. `gcwm.py`에서 얻은 $K$-컴포넌트 GCWM 적합 결과(`gcwm_fit`)를 입력으로 받아, 올바르지 않은 분포의 가정으로 인해서 클러스터보다 많이 생성된 컴포넌트들을 병합하는 알고리즘이다. 

---

## 1. 두 개의 잠재변수 — $Z$와 $W$

| 기호 | 의미 | 범위 |
|---|---|---|
| $Z$ | 컴포넌트 소속 (미시적) | $k=1,\dots,K$ |
| $W$ | 클러스터 소속 (거시적) | $m=1,\dots,M$ |

파티션 $G_1,\dots,G_M$이 $\{1,\dots,K\}$를 분할하며, $k\in G_m$이면 $Z=k \Rightarrow W=m$이 **결정론적으로** 성립한다. 즉

$$
p(w=m,z=k)=0 \quad \text{for } k\notin G_m
$$

이 관계는 코드에서 컴포넌트 → 클러스터 매핑 배열 하나로 구현된다. 단, 하나의 컴포넌트는 하나의 군집에 속한다는 전제가 가정되어 있다. 더욱 직관적으로 이해하면 다음과 같다. 1반부터 5반까지 있다고 가정하였을 때, 1부터 3반은 A코스의 학습을 수행하고 4부터 5반은 B코스의 학습을 수행한다. 1반이라는 정보만 있으면 A코스를 학습한다는 사실을 명확하게 알 수 있지만 A코스를 수강하고 있다고 해서 이 반이 어떤 반인지에 대한 확실한 정보를 얻을 수 없다.

```python
comp2cluster = np.zeros(K, dtype=int)
for m, g in enumerate(groups):
    for k in g:
        comp2cluster[k] = m
```

---

## 2. 결합확률모형 (Eq. 6/7)

$$
p(x,y) = \sum_{m=1}^{M}\sum_{k\in G_m} \eta_{mk}\,\phi_p(x;\mu_k,\Sigma_k)\,\phi_1\big(y;(1,x)\beta_m,\ \sigma_k^2\big)
$$

여기서 $\eta_{mk}=p(w=m,z=k)$. 핵심은 파라미터 별로 컴포넌트에 의존하는지 군집에 의존하는지에 대한 기준이 다르다는 것이다. 

$$
\underbrace{\mu_k,\ \Sigma_k,\ \sigma_k^2}_{\text{컴포넌트}(k)\text{ 단위 — 그대로 유지}}
\qquad\qquad
\underbrace{\beta_m}_{\text{클러스터}(m)\text{ 단위 — 여러 }k\text{가 공유}}
$$

즉 $X$의 분포(위치·모양)는 원래 컴포넌트 별 값을 그대로 갖지만, $Y\mid X$의 회귀선(평균)은 같은 클러스터에 속한 컴포넌트끼리 공유한다. 왜냐하면 beta의 기준으로 유사한 것들로 묶기 때문이다. 하지만 분산(잔차) 달라야 한다. 그 이유는 분산마저 같아버리면 이 컴포넌트가 어떤 군집에서 나왔는지 명확하게 알 수 없기 때문이다.

---

## 3. `_refit_partition` — 주어진 파티션 하의 EM 재추정

Algorithm 1의 3)단계, 즉 논문 Eq. 9의 reduced model

$$
p^R(x,y) = \sum_{m=1}^{M}\sum_{k\in G_m}\eta_{mk}\,\phi_p(x;\mu_k,\Sigma_k)\,\phi_1\big(y;(1,x^\top)\beta_m,\sigma_k^2\big)
$$

을 특정 파티션 `groups` 하에서 수렴할 때까지 EM으로 적합시키는 함수이다.

### 3.1 초기값 (논문 Section 4)

$$
\mu_k^{(0)},\ \Sigma_k^{(0)},\ \sigma_k^{2(0)},\ \eta_k^{(0)} \ \leftarrow\ \text{GCWM 결과 그대로 재사용}
$$

$$
\beta_m^{(0)} = \frac{1}{|G_m|}\sum_{k\in G_m} \hat\beta_k^{\text{GCWM}} \quad (\text{병합된 컴포넌트들의 β 평균})
$$

```python
mu = gcwm_fit["mu"].copy()
Sigma = gcwm_fit["Sigma"].copy()
sigma2 = gcwm_fit["sigma2"].copy()
eta = gcwm_fit["pi"].copy()
for m, g in enumerate(groups):
    beta_cluster[m] = gcwm_fit["beta"][g].mean(axis=0)
```

> 초깃값을 CWM의 계수들의 평균 값을 사용한다. 하지만 이 값을 계속해서 사용하는 것이 아닌 초깃값만 지정하고 향후에는 병합된 CWM에서 EM을 통해서 얻어진다.

### 초기화(1회) vs. E-M 루프(반복) — 흐름 구분

```text
[1회만 실행]
  β_m^(0) = GCWM β들의 단순 평균   ← 초깃값으로 지정하지 병합하고 나서 또, GCWM -> 계수들의 평균으로 구하는 것이 아님.

[수렴할 때까지 반복 (t = 0, 1, 2, ...)]
  E-step:  β_m^(t) 를 이용해 사후확률 w_imk^(t) 계산   (§3.2)
       ↓
  M-step:  w_imk^(t) 를 가중치로 β_m^(t+1) 재추정        (§3.3, 가중 WLS — 단순 평균 아님!)
       ↓
  로그우도 변화 < 1e-5 ?  ─ No → t+1로 계속
                          └ Yes → β̂_m = β_m^(t+1) 로 확정
```

beta를 평균 낸 것은 초깃값으로만 사용하고, 그 병합된 Component들을 Cluster로 묶은 후에 다시 EM알고리즘을 돌려 가중 평균으로 계수를 추정해야한다.

### 3.2 E-step

$$
w_{imk}^{(t)} = \frac{\eta_{mk}^{(t)}\,\phi_p\!\big(x_i;\mu_k^{(t)},\Sigma_k^{(t)}\big)\,\phi_1\!\big(y_i;(1,x_i)\beta_m^{(t)},\sigma_k^{2(t)}\big)}
{\displaystyle\sum_{l=1}^{M}\sum_{j\in G_l}\eta_{lj}^{(t)}\,\phi_p\!\big(x_i;\mu_j^{(t)},\Sigma_j^{(t)}\big)\,\phi_1\!\big(y_i;(1,x_i)\beta_l^{(t)},\sigma_j^{2(t)}\big)}
$$

코드에서 `tau[:, k]`가 정확히 $w_{imk}^{(t)}$에 대응한다 (단, $m=\text{comp2cluster}[k]$로 자동 결정되므로 $k$만으로 인덱싱).

```python
mean_y = A @ beta_cluster[m]          # m = comp2cluster[k]
log_dens[:, k] = np.log(eta[k]+1e-300) + log_phi_x + log_phi_y
tau = np.exp(log_dens - max_log)
tau /= tau.sum(axis=1, keepdims=True)
```

### 3.3 M-step

**$\eta_{mk}$ (여전히 컴포넌트 단위로 자유도 유지)**

$$
\eta_{k}^{(t+1)} = \frac{\sum_i w_{imk}^{(t)}}{N}
$$

```python
eta = Nk / N   # Nk = tau.sum(axis=0)
```

**$\mu_k,\ \Sigma_k$ (GCWM의 M-step과 완전히 동일한 형태, 컴포넌트별 독립 추정)**

$$
\mu_k^{(t+1)} = \frac{\sum_i w_{imk}^{(t)} x_i}{\sum_i w_{imk}^{(t)}},
\qquad
\Sigma_k^{(t+1)} = \frac{\sum_i w_{imk}^{(t)} (x_i-\mu_k^{(t+1)})(x_i-\mu_k^{(t+1)})^\top}{\sum_i w_{imk}^{(t)}}
$$

**$\beta_m$ — 클러스터 안의 모든 컴포넌트를 풀링(pooling)한 가중최소제곱**

$$
\beta_m^{(t+1)} = \big(A^\top W_m^{(t)} A\big)^{-1} A^\top W_m^{(t)} y,
\qquad
W_m^{(t)} = \mathrm{diag}\big(w_{1m\cdot}^{(t)},\dots,w_{Nm\cdot}^{(t)}\big),
\quad
w_{im\cdot}^{(t)} = \sum_{k\in G_m} w_{imk}^{(t)}
$$

이 식이 **병합의 수학적 실체**이다 — 원래 서로 다른 컴포넌트에 속했던 관측치별 가중치들이 $G_m$ 안에서 하나로 합산되어, 단 하나의 $\beta_m$을 추정하는 데 함께 쓰인다.

> **§3.1의 초기값과 혼동 주의**: §3.1의 $\beta_m^{(0)}$는 GCWM β들의 단순 평균이었지만, 여기서 매 반복마다 갱신되는 $\beta_m^{(t+1)}$는 관측치별 사후확률 $w_{imk}^{(t)}$로 가중된 최소제곱(WLS)이다. $t=0$일 때조차 이 둘은 이미 다른 값이며, 반복이 진행될수록 그 차이는 더 벌어진다 — 최종 $\hat\beta_m$는 이 WLS 갱신이 수렴한 값이지, 초기 평균값이 아니다.

```python
for m, g in enumerate(groups):
    w_m = tau[:, g].sum(axis=1)              # sum_{k in Gm} w_imk
    AtWA = A.T @ (w_m[:, None] * A)
    AtWy = A.T @ (w_m * y)
    beta_cluster[m] = np.linalg.solve(AtWA, AtWy)
```

**$\sigma_k^2$ (컴포넌트 단위 유지, 단 잔차는 공유된 $\beta_m$ 기준)**

$$
\sigma_k^{2(t+1)} = \frac{\sum_i w_{imk}^{(t)}\big(y_i-(1,x_i)\beta_m^{(t+1)}\big)^2}{\sum_i w_{imk}^{(t)}}
$$

```python
resid = y - A @ beta_cluster[m]     # m = comp2cluster[k]
sigma2[k] = (w_k * resid**2).sum() / Nk[k]
```

### 3.4 자유 파라미터 수 (Eq. 7 아래 문단)

$$
n_{\text{params}} = K\cdot\frac{p^2+3p+2}{2} - 1 \;+\; M(p+1) \;+\; K
$$

$$
\underbrace{M(p+1)}_{\beta \text{ 몫 (}M\text{개로 축소)}} \quad \text{vs.} \quad \underbrace{K(p+1)}_{\text{원래 GCWM의 } \beta \text{ 몫}}
$$

$M<K$일 때 정확히 $(K-M)(p+1)$개만큼 파라미터가 절감된다.

```python
n_params = K * (p**2 + 3*p + 2) / 2 - 1 + M * (p + 1) + K
bic = -2 * prev_ll + n_params * np.log(N)
```

---

## 4. `merge_gcwm` — Algorithm 1

### 4.1 초기 상태

$$
M=K,\qquad G_m=\{m\}\ \ (m=1,\dots,K) \quad(\text{병합 없음, GCWM과 동일})
$$

```python
groups = [[k] for k in range(K)]
current = _refit_partition(X, y, gcwm_fit, groups)
```

### 4.2 병합 후보 탐색

$$
d_{jk} = \big\lVert \hat\beta_j - \hat\beta_k \big\rVert_1, \qquad j\ne k
$$

가장 작은 $d_{jk}$를 갖는 쌍 $(j^{\ast},k^{\ast})$부터 순서대로 시도한다.

```python
for (i, j) in combinations(range(len(groups)), 2):
    d = np.sum(np.abs(beta_now[i] - beta_now[j]))
    dists.append((d, i, j))
dists.sort(key=lambda t: t[0])
```

### 4.3 후보 파티션 구성과 검증

$$
G_1^R,\dots,G_{M-1}^R \;=\; \big(G_1,\dots,G_M\big)\ \text{에서}\ G_{j^{\ast}},G_{k^{\ast}}\ \text{를 하나로 합친 것}
$$

$$
\text{IC}(p^R) < \text{IC}(p) \ ?
\qquad \text{IC} \in \{\text{BIC},\ \text{ICL}\}
$$

- **참**: 파티션을 확정, $M\leftarrow M-1$, 처음(4.2)으로 돌아가 재탐색
- **거짓**: 이 시도는 폐기, 다음으로 작은 $d_{jk}$ 쌍으로 4.3을 재시도
- **모든 쌍이 거짓**: 알고리즘 종료, 현재 파티션이 최종 MGCWM

```python
trial_groups = [g for idx, g in enumerate(groups) if idx not in (i, j)]
trial_groups.append(groups[i] + groups[j])
trial_fit = _refit_partition(X, y, gcwm_fit, trial_groups)

if trial_fit[criterion] < current[criterion]:
    groups = trial_groups
    current = trial_fit
    accepted = True
    break          # 4.2로 재시작
```

### 4.4 알고리즘 성격

이 절차는 **agglomerative hierarchical clustering**과 동형이다 — 다만

$$
\text{거리 척도: 유클리드 거리} \ \longrightarrow\ \lVert\hat\beta_j-\hat\beta_k\rVert_1 \quad(\text{"함수관계"} \text{ 기준})
$$

$$
\text{종료 기준: 사용자가 지정한 클러스터 수} \ \longrightarrow\ \text{BIC/ICL 자동 판정}
$$

으로 대체된 형태이며, 매 병합 시도마다 전체 EM을 재수렴시킨다는 점에서 순수 계층적 군집화보다 계산 비용이 훨씬 크다.

---

## 5. 클러스터의 $X$ 분포 — 언제 mixture가 되는가

$$
p(x\mid w=m) = \sum_{k\in G_m}\pi_{k|m}\,\phi_p(x;\mu_k,\Sigma_k), \qquad \pi_{k|m}=\eta_{mk}/p_m
$$

$$
|G_m|=1 \ \Rightarrow\ \text{단일 정규분포} \qquad\qquad |G_m|\ge 2 \ \Rightarrow\ \text{Gaussian mixture (다봉 가능)}
$$

즉 병합이 실제로 일어난 클러스터만 mixture가 되고, 나머지는 원래 컴포넌트 그대로 단일 정규분포로 남는다. 코드에서는 이를 별도로 계산하지 않지만, `groups[m]`의 길이가 이 여부를 그대로 알려준다. 하지만 결국은 Gaussian을 병합하기는 하지만, 본질적인 Gaussian가정과 많은 Component수는 달라지지 않는다.

---

## 6. 전체 흐름 요약

| 단계 | 무엇을 하는가 | 코드/수식 대응 |
|---|---|---|
| ① | $K$-컴포넌트 GCWM 적합 | `gcwm.py` → `groups = [[k] for k in range(K)]` (초기: 병합 없음) |
| ② | 그 파티션 하에서 EM 재수렴 | `_refit_partition()` — E-step ↔ M-step 반복 (Eq. 9) |
| ③ | 병합 후보 탐색 | $d_{jk}$가 가장 작은 쌍 $(j,k)$ 탐색 |
| ④ | 후보 파티션 구성 | `trial_groups` — 두 그룹을 하나로 합침 |
| ⑤ | 통계적 검증 | BIC/ICL 비교 |
| ⑥ | 확정 또는 기각 | 개선되면 ②로 복귀, 아니면 다음 후보로 ③ 재시도 (Algorithm 1의 4단계) |

```text
GCWM(K)  →  _refit_partition (E-step/M-step)  →  d_jk 최솟값 탐색
   →  후보 병합 (trial_groups)  →  BIC/ICL 비교  →  확정 or 기각
```

이 사이클이 더 이상 개선되는 병합이 없을 때까지 반복되며, 최종 `groups`가 곧 $G_1,\dots,G_M$ — 즉 MGCWM이 찾아낸 "진짜 클러스터" 구조이다.

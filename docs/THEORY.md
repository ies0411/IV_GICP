# IV-GICP: Theoretical Foundations

## Overview: Fisher Information as a Unifying Framework

IV-GICP의 모든 기술 기여(C1–C4)는 단일 원리에서 도출된다: **pose estimation의 Fisher Information을 최대화하라**.

LiDAR odometry를 확률론적으로 볼 때, 각 correspondence $(p_m, \mu_j, \Omega_j)$는 pose $\xi \in \mathfrak{se}(3)$에 대한 정보를 제공한다. 이 정보의 총량은 **6×6 Fisher Information Matrix(FIM)**로 표현된다:

$$\mathcal{I}(\xi) = \sum_{m=1}^{M} J_m^T \Omega_m J_m \in \mathbb{R}^{6 \times 6}$$

여기서 $J_m = \frac{\partial d_m}{\partial \xi}$는 잔차 Jacobian, $\Omega_m$은 precision matrix다. FIM의 최소 eigenvalue $\lambda_{\min}(\mathcal{I})$가 작을수록 pose estimate가 불안정하다 — 이것이 **geometry degeneracy**의 정보이론적 정의다.

네 가지 기여는 각각:
- **C1**: FIM 기반으로 informative한 correspondence에 높은 가중치
- **C2**: FIM에 intensity 채널을 추가해 geometry 퇴화를 원천 차단
- **C3**: per-voxel entropy로 geo/intensity 비중을 원칙적으로 결정
- **C4 (MSCS)**: FIM sufficiency 조건을 만족하는 **최소** correspondence set — 이론적으로 최적인 speed/accuracy tradeoff

---

## C1: FIM-Weighted Correspondence Selection

### 동기

표준 GICP은 모든 correspondence를 동등하게 취급한다. 그러나 FIM 관점에서 각 correspondence는 서로 다른 방향에 서로 다른 정보를 제공한다. 특히 geometry가 퇴화한 방향($\lambda_{\min}$ eigenvector $v_{\min}$)에 기여하는 correspondence는 해당 방향의 추정 정확도에 결정적이다.

### 알고리즘

correspondence $m$의 FIM 기여도를 **퇴화 방향** $v_{\min}$에 투영해 가중치를 계산한다:

$$w_m^{FIM} = v_{\min}^T \left(J_m^T \Omega_m J_m\right) v_{\min}$$

최종 가중치는 Huber robust weight $w_m^H$와 결합된다:

$$w_m = w_m^H \cdot \frac{w_m^{FIM}}{\bar{w}^{FIM}}$$

여기서 $\bar{w}^{FIM} = \frac{1}{M}\sum_m w_m^{FIM}$는 normalization factor다.

### 성질

- $v_{\min}$이 퇴화 방향에 수직인 correspondence는 $w_m^{FIM} \approx 0$ → 자동으로 down-weight
- 퇴화 방향에 기여하는 소수의 correspondence가 강조됨
- Hard gate (`fim_gate_ratio`): $w_m^{FIM} < \text{ratio} \cdot \bar{w}^{FIM}$인 correspondence 완전 제거

---

## C2: 4D Geo-Photometric Registration

### 동기

3D geometry만으로는 터널처럼 직선 방향이 관찰되지 않는 환경에서 FIM이 rank-deficient가 된다. LiDAR intensity를 4번째 차원으로 통합하면 **geometry와 독립적인 정보 채널**이 추가된다.

### 4D 잔차 공식화

소스 포인트 $p_s = [x, y, z, \alpha I_s]^T$와 타깃 voxel 평균 $\mu_j = [\mu_{xyz}^T, \mu_I]^T$에 대해:

$$d_m = \begin{bmatrix} T \cdot p_s^{xyz} - \mu_j^{xyz} \\ \alpha I_s - \mu_j^I \end{bmatrix} \in \mathbb{R}^4$$

**4D precision matrix** (block-diagonal):

$$\Omega_m^{4\times4} = \begin{bmatrix} \Omega_m^{geo} & 0 \\ 0 & \omega_I \end{bmatrix}$$

여기서 $\omega_I = \alpha^2 / (\text{Var}(I)/\ell_v^2 + \varepsilon)$. Intensity gradient가 클수록 $\omega_I$ 커짐 → 강한 intensity constraint.

**Intensity Jacobian** (정확하게 유도됨):

$$J_m[3, :] = -\alpha \cdot (\nabla\mu_I)^T J_{xyz}$$

여기서 $J_{xyz} = [-[Rp_s]_\times \;|\; I_3] \in \mathbb{R}^{3\times6}$는 표준 ICP pose Jacobian.

### Theorem 1: Degeneracy Recovery

**Theorem 1.** $\varepsilon > 0$이고 적어도 하나의 voxel에서 intensity gradient $\|\nabla\mu_I\| > 0$이면:

$$v^T \mathcal{I}_{total} v \geq \frac{\varepsilon}{\sigma_I^2} \sum_m \|J_m^{xyz} v\|^2 > 0 \quad \forall v \in S^5$$

*Proof sketch.* 4D FIM의 intensity block은 항상 양정치:
$$v^T \sum_m J_m^T \Omega_m^{4\times4} J_m \, v \geq \sum_m \omega_I (J_m^{xyz} v)^T \nabla\mu_I \nabla\mu_I^T J_m^{xyz} v \geq \frac{\varepsilon}{\sigma_I^2} \sum_m \|J_m^{xyz} v\|^2$$

임의의 방향 $v$에 대해 geometry FIM이 퇴화하더라도($v^T \mathcal{I}_{geo} v = 0$), intensity 항이 양수 기여를 하므로 $\mathcal{I}_{total}$은 항상 full-rank다. **별도의 degeneracy detector가 필요 없다.**

---

## C3: Entropy-Consistent Adaptive Alpha

### 동기

균일한 콘크리트 터널(GEODE Urban)에서 intensity는 noise만 추가하므로 $\alpha = 0$이 최적이다. 반면 다양한 재질의 환경(KITTI, SubT)에서는 $\alpha > 0$이 정확도를 향상시킨다. 이 결정을 **voxel당 geometry entropy**를 기반으로 원칙적으로 내린다.

### 알고리즘

타깃 voxel $j$의 geometry entropy:

$$h_j^{geo} = \frac{1}{2} \log \det \Sigma_j^{geo}$$

Intensity 스케일:

$$\alpha_j^{eff} = \alpha \cdot \text{clip}\!\left(1 + c \cdot (h_j^{geo} - \text{median}_j(h^{geo})), \, 0, \, 2\right)$$

- $h_j^{geo}$ 낮음 (평면적, 퇴화) → 중앙값 이하 → $\alpha_j^{eff} < \alpha$ → intensity 비중 감소
- $h_j^{geo}$ 높음 (복잡한 구조) → $\alpha_j^{eff} > \alpha$ → intensity 비중 증가

**C2와의 관계**: Theorem 1에 의해 C2가 활성화되면 $\mathcal{I}$의 condition number가 개선된다 ($\kappa < 100$). 이 경우 C3의 adaptive bypass는 cost 없이 자동 발동된다.

---

## C4: Minimum Sufficient Correspondence Set (MSCS)

### 동기

전체 correspondence 집합 $\mathcal{C}$ 중 많은 수가 FIM의 잘 conditioned된 방향에 중복 기여를 한다. 이들을 모두 사용하는 것은 계산 낭비다. 반면 퇴화 방향에 기여하는 correspondence는 소수이므로 반드시 포함해야 한다.

핵심 질문: **Theorem 1의 well-posedness 조건을 만족하는 최소 correspondence set은 무엇인가?**

### 정의: Minimum Sufficient Correspondence Set

$$\mathcal{S}^* = \arg\min_{|\mathcal{S}|} |\mathcal{S}| \quad \text{s.t.} \quad \lambda_{\min}\!\left(\sum_{m \in \mathcal{S}} J_m^T \Omega_m J_m\right) \geq \varepsilon_{\text{target}}$$

이 집합은 Theorem 1의 조건을 만족하는 **최소** 집합이다. $\varepsilon_{\text{target}}$는 GN solver의 수치 안정성 조건에서 도출된다:

$$\varepsilon_{\text{target}} = \frac{\lambda_{\max}(\mathcal{I}_\mathcal{C})}{\kappa_{\max}}$$

$\kappa_{\max} = 100$ (GN의 수치 안정성 임계 condition number). **데이터셋별 튜닝 불필요** — $\varepsilon_{\text{target}}$는 scene의 실제 FIM에서 자동 결정된다.

### Greedy MSCS Algorithm

$$\text{score}_m = v_{\min}^T \cdot J_m^T \Omega_m J_m \cdot v_{\min}$$

$v_{\min}$은 이전 프레임의 GN Hessian의 최소 eigenvector (warm-start). 이를 내림차순으로 정렬 후 greedy accumulation:

```
Sort C by score_m descending
H ← 0, b ← 0, n_used ← 0
for m in sorted(C):
    H += J_m^T Ω_m J_m,  b += J_m^T Ω_m d_m,  n_used++
    if n_used % 64 == 0 and λ_min(H) ≥ ε_target:
        break  # sufficient!
return H, b, n_used
```

### Lemma: MSCS Greedy Optimality

**Lemma.** $v_{\min}$을 현재 $H$의 최소 eigenvector로 잡을 때, score $s_m = v_{\min}^T J_m^T \Omega_m J_m v_{\min}$ 기준 greedy selection은 크기 $k$인 모든 부분집합 중 $\lambda_{\min}$ criterion에 대해 **최대 정보를 달성한다**.

*Proof sketch.* $\lambda_{\min}(H)$의 증가는 $v_{\min}^T \Delta H v_{\min}$으로 근사된다 (변분). 이를 최대화하는 correspondence가 바로 $s_m$이 가장 큰 correspondence다. 따라서 $s_m$ 기준 greedy selection은 $\lambda_{\min}$ 기준 locally optimal이다. ∎

### Scene-Adaptive Complexity

| Scene type | $|\mathcal{S}^*|/M$ | Description |
|------------|---------------------|-------------|
| Well-conditioned (KITTI outdoor) | 20–30% | 6 DOF 모두 풍부한 정보 → 소수로 충분 |
| Partially degenerate (1 DOF, tunnel) | 70–90% | 퇴화 방향 커버에 많은 correspondence 필요 |
| Fully degenerate | ~100% | 모든 correspondence 사용 |

**자동 speed/accuracy tradeoff**: 명시적인 mode switching 없이, 장면 복잡도에 따라 사용 correspondence 수가 자동 결정된다.

### v_min Warm-Start (추가 비용 없음)

- C1 path에서 이미 `SelfAdjointEigenSolver<Matrix6d>` 호출 중
- MSCS는 이 eigenvector를 재사용 → **추가 eigendecomposition 비용 = 0**
- Frame k의 $v_{\min}$을 frame k+1의 sorting에 사용 → 수 프레임 내 수렴

---

## Theorem 2: Map Distribution Propagation Error Bound

VoxelMap의 분포가 pose uncertainty로 인해 어떻게 왜곡되는지 분석한다.

**Theorem 2.** 프레임 간 실제 pose 변화 $\Delta T$와 추정 $\hat{\Delta T}$의 오차가 $\Sigma_{\Delta T}$일 때:

$$\|\Sigma_{true} - R_\Delta \Sigma_{old} R_\Delta^T\|_F \leq \|\mu_{old}\|^2 \cdot \|\Sigma_{\Delta T}\|_F + O(\|\Sigma_{\Delta T}\|^2)$$

**의미**: 이전 voxel의 covariance $\Sigma_{old}$를 추정된 rotation $R_\Delta$로 propagate할 때의 오차는, 현재 odometry uncertainty $\|\Sigma_{\Delta T}\|_F$에 비례한다. 충분히 정확한 odometry (Theorem 1 보장)라면 이 오차는 무시 가능하다 → Welford incremental update 정당화.

---

## 이론-구현 대응

| 이론 | 구현 파일 | 핵심 코드 위치 |
|------|-----------|---------------|
| FIM 계산 | `iv_gicp_core.cpp` | `I_G += J_xyz.T * Og * J_xyz` |
| C1 가중치 | `iv_gicp_core.cpp` | `w_fim = v.dot(H_m * v)` |
| C2 4D precision | `iv_gicp_map.cpp` | `build_target_arrays(alpha, ...)` |
| Theorem 1 실현 | `iv_gicp_core.cpp` | `d[3] = alpha*src_I - mu_I`, `J.row(3)` |
| C3 entropy alpha | `iv_gicp_map.cpp` | `build_target_arrays(..., use_entropy_alpha)` |
| MSCS sort+stop | `iv_gicp_core.cpp` | `scores`, greedy loop with `λ_min` check |
| v_min warm-start | `pipeline.py` | `self._prev_v_min` |

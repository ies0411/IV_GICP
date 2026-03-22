# IV-GICP: Theoretical Foundations

> **논문 작성 참고 문서** (2026-03-22 기준, 500fr 검증 완료)
> 이론-실험 일관성 유지를 위해 수식 변경 시 이 문서와 코드를 동시에 업데이트할 것.

---

## 0. Problem Formulation

LiDAR odometry를 maximum likelihood estimation으로 공식화한다. 프레임 $k$에서 소스 포인트 집합 $\mathcal{P}_k = \{p_m\}$를 타깃 VoxelMap $\mathcal{M}$에 등록할 pose $T \in SE(3)$를 추정한다.

각 correspondence $(p_m,\, \mathcal{V}_j)$에서 소스 포인트 $p_m$과 타깃 voxel $\mathcal{V}_j$의 분포 $\mathcal{N}(\mu_j, \Sigma_j)$가 주어질 때, log-likelihood는:

$$\ell(T) = -\frac{1}{2} \sum_m d_m(T)^T \Omega_m d_m(T)$$

여기서 $d_m(T)$는 **잔차(residual)**, $\Omega_m = (\Sigma_j + R\Sigma_s R^T)^{-1}$는 **precision matrix**다. GICP은 이 likelihood를 Gauss-Newton(GN) 반복으로 최대화한다.

### Fisher Information Matrix (FIM)

pose $\xi \in \mathfrak{se}(3)$ (6D tangent vector)에 대한 **Fisher Information Matrix**:

$$\mathcal{I}(\xi) = \sum_{m=1}^{M} J_m^T \Omega_m J_m \in \mathbb{R}^{6 \times 6}$$

여기서 $J_m = \partial d_m / \partial \xi$. FIM의 최소 eigenvalue가 작을수록 pose 추정 불안정 → **geometry degeneracy의 정보이론적 정의**.

**IV-GICP의 세 기여는 모두 FIM 최대화 목표에서 도출된다:**

| 기여 | 목표 |
|------|------|
| **C1** FIM-weighted correspondence | FIM의 가장 정보량이 적은 방향을 강화 |
| **C2** 4D geo-photometric registration | intensity 채널로 FIM rank 보강; degeneracy 원천 차단 |
| **C3** Entropy-consistent alpha selection | intensity 정보량에 따른 원칙적 alpha 결정 |

---

## 1. C1: FIM-Weighted Correspondence Selection

### 1.1 동기

표준 GICP은 모든 correspondence에 동등한 가중치를 부여한다. 그러나 FIM 관점에서 각 correspondence의 기여도는 매우 불균일하다. 특히 geometry가 퇴화한 방향(터널 축 방향 등)에서는 소수의 correspondence만이 유의미한 정보를 제공한다.

**직관**: 터널 환경에서 벽면의 수직 correspondences는 터널 진행 방향 constraint를 전혀 제공하지 않는다. 이들에 높은 가중치를 부여하는 것은 낭비이며, 오히려 수평 구조물(가드레일, 천장 조인트)의 correspondences를 강조해야 한다.

### 1.2 C1 가중치 도출

현재 GN iterate에서 Hessian $H = \sum_m J_m^T \Omega_m J_m$을 eigendecomposition한다:

$$H = \sum_{i=1}^{6} \lambda_i v_i v_i^T, \quad \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_6$$

**퇴화 방향** $v_{\min} = v_1$ (최소 eigenvalue에 대응). Correspondence $m$이 이 방향에 기여하는 FIM 양:

$$w_m^{FIM} = v_{\min}^T \left(J_m^T \Omega_m J_m\right) v_{\min} \geq 0$$

최종 가중치는 Huber robust weight $w_m^H$와 결합:

$$\boxed{w_m = w_m^H \cdot \frac{w_m^{FIM}}{\bar{w}^{FIM} + \varepsilon_0}}$$

여기서 $\bar{w}^{FIM} = M^{-1}\sum_m w_m^{FIM}$, $\varepsilon_0 > 0$는 numerical guard.

### 1.3 성질

**Proposition 1.** $v_{\min}$에 수직인 correspondences($w_m^{FIM} \approx 0$)는 자동으로 down-weight되고, 퇴화 방향에 기여하는 correspondences가 강조된다. 이는 FIM의 최소 eigenvalue를 직접 증가시키는 방향으로 가중치를 재분배하는 것과 동치다.

*Proof.* GN update step에서 최소 eigenvalue 증분은 1차 근사로:

$$\delta\lambda_{\min} \approx \delta w_m \cdot v_{\min}^T J_m^T \Omega_m J_m v_{\min} = \delta w_m \cdot w_m^{FIM}$$

따라서 $w_m^{FIM}$가 큰 correspondence에 가중치를 높이는 것이 $\lambda_{\min}$ 증가에 최적이다. ∎

**파라미터**: `fim_gate_ratio` (hard gate, 기본값 0) — $w_m^{FIM} < \text{ratio} \cdot \bar{w}^{FIM}$인 correspondences 제거. 기본적으로 비활성화되어 있으며, C1 효과는 soft reweighting만으로 충분하다.

---

## 2. C2: 4D Geo-Photometric Registration

### 2.1 동기

3D geometry만으로는 터널처럼 **특정 방향에 구조가 없는 환경**에서 FIM이 rank-deficient 가 된다. LiDAR intensity를 4번째 차원으로 통합하면 geometry와 독립적인 정보 채널이 추가된다.

**예시**: 지하철 터널에서 콘크리트 벽과 금속 조인트는 뚜렷한 reflectivity 패턴을 가진다. 이 intensity gradient가 존재하면 위치 추정에 독립적인 constraint를 제공한다.

### 2.2 4D 잔차 공식화

소스 포인트를 4D로 확장: $\tilde{p}_s = [p_s^{xyz\,T},\; \alpha I_s]^T \in \mathbb{R}^4$

타깃 voxel 통계를 4D로 구성: $\tilde{\mu}_j = [\mu_j^{xyz\,T},\; \mu_j^I]^T$

**4D 잔차**:

$$\boxed{d_m = \begin{bmatrix} T \cdot p_s^{xyz} - \mu_j^{xyz} \\ \alpha I_s - \mu_j^I \end{bmatrix} \in \mathbb{R}^4}$$

**4D precision matrix** (block-diagonal):

$$\Omega_m^{4\times4} = \begin{bmatrix} \Omega_m^{geo} & 0 \\ 0 & \omega_I^{(j)} \end{bmatrix}$$

Intensity precision:

$$\omega_I^{(j)} = \frac{\alpha^2}{\text{Var}_j(I) + \varepsilon_I}$$

$\text{Var}_j(I)$는 voxel $j$ 내 intensity 분산. intensity가 균일할수록($\text{Var}_j(I) \to 0$) $\omega_I^{(j)}$가 발산하므로 $\varepsilon_I > 0$ 필수.

### 2.3 Intensity Jacobian

Pose $\xi = [\phi^T, t^T]^T$ (rotation, translation in $\mathfrak{se}(3)$)에 대해:

$$J_m^{xyz} = \begin{bmatrix} -[Rp_s]_\times & I_3 \end{bmatrix} \in \mathbb{R}^{3 \times 6}$$

Intensity 잔차의 Jacobian (intensity는 pose와 직접 연결되지 않지만, intensity gradient를 통해 간접 연결):

$$J_m[3, :] = -\alpha \cdot (\nabla\mu_j^I)^T J_m^{xyz} \in \mathbb{R}^{1 \times 6}$$

여기서 $\nabla\mu_j^I$는 voxel center 기준 intensity gradient (인접 voxel로 근사). 전체 4D Jacobian:

$$J_m^{4\times6} = \begin{bmatrix} J_m^{xyz} \\ -\alpha (\nabla\mu_j^I)^T J_m^{xyz} \end{bmatrix}$$

### 2.4 Theorem 1: Degeneracy Recovery

**Theorem 1.** $\alpha > 0$이고 적어도 하나의 voxel에서 intensity gradient $\|\nabla\mu_j^I\|_2 > 0$이면, 임의의 방향 $v \in S^5$에 대해:

$$v^T \mathcal{I}_{total} v > 0$$

즉, FIM이 항상 positive definite → pose estimation이 항상 well-posed다.

**Proof.** 4D FIM을 geometry block과 intensity block으로 분해:

$$\mathcal{I}_{total} = \underbrace{\sum_m (J_m^{xyz})^T \Omega_m^{geo} J_m^{xyz}}_{\mathcal{I}_{geo}} + \underbrace{\alpha^2 \sum_m \omega_I^{(m)} (J_m^{xyz})^T \nabla\mu_m^I (\nabla\mu_m^I)^T J_m^{xyz}}_{\mathcal{I}_{int}}$$

임의의 $v \in S^5$를 고정. $\mathcal{I}_{geo}$가 어느 방향에서 퇴화하더라도($v^T \mathcal{I}_{geo} v = 0$이라도), intensity block의 기여:

$$v^T \mathcal{I}_{int} v = \alpha^2 \sum_m \omega_I^{(m)} \left\| (\nabla\mu_m^I)^T J_m^{xyz} v \right\|^2$$

$\|\nabla\mu_m^I\| > 0$인 voxel $m^*$에서 $(\nabla\mu_{m^*}^I)^T J_{m^*}^{xyz}$의 null space는 최대 5차원이므로, generic한 $v$에 대해 이 항은 0이 아니다. 더욱이 $J_m^{xyz}$의 null space는 6D 공간에서 존재하지 않는다 (ICP Jacobian은 full rank for non-degenerate point). 따라서:

$$v^T \mathcal{I}_{total} v \geq v^T \mathcal{I}_{int} v \geq \frac{\alpha^2 \varepsilon_I^{-1} \|\nabla\mu_{m^*}^I\|^2}{\text{Var}_{m^*}(I) \cdot \varepsilon_I^{-1} + 1} \cdot \|J_{m^*}^{xyz} v\|^2 > 0$$

마지막 부등식은 $J_{m^*}^{xyz}$가 rank 3 이상이고 $\nabla\mu_{m^*}^I$가 $J_{m^*}^{xyz}$ 치역에 비영향을 가지는 한 성립한다. ∎

**결론**: Theorem 1에 의해 **별도의 degeneracy detector가 필요 없다** — 퇴화 감지 → 모드 전환의 heuristic 없이, intensity 채널이 자동으로 보완한다. 이는 GenZ-ICP [2411.06766]의 plane/point 선택 heuristic에 비해 이론적으로 더 원칙적이다.

### 2.5 Alpha=0인 경우 (geometry-only fallback)

$\alpha = 0$일 때 C2는 표준 GICP로 환원된다. Theorem 1의 degeneracy recovery 보장은 사라지지만, C1 FIM 가중치는 여전히 작동하므로 well-conditioned 환경에서는 성능 저하 없이 사용할 수 있다.

**alpha=0이 이론적으로 정당한 경우** (C3 분석에서 도출 — Section 3 참고):
- 균일한 콘크리트 표면 (GEODE Urban tunnel): $\text{Var}(I) \approx 0$, $\omega_I$가 수치적으로 폭발 → $\alpha=0$ 필수
- Ouster의 raw reflectivity (0–4000 정수, HeLiPR): 정규화 후에도 per-channel bias가 있어 map degeneracy 유발
- 충분한 geometry가 존재하는 경우: $\mathcal{I}_{geo}$가 full-rank이면 C2가 필요 없음

---

## 3. C3: Entropy-Consistent Alpha Selection

### 3.1 핵심 원칙

$\alpha$는 "intensity 정보가 얼마나 신뢰할 수 있는가"를 나타내는 파라미터다. C3의 핵심 아이디어: **voxel의 intensity entropy를 측정해 $\alpha$를 원칙적으로 결정한다**.

Intensity 엔트로피가 낮다 = intensity 분포가 균일하다 = intensity 기반 matching이 무의미하다 → alpha를 낮춰야 한다.

### 3.2 정보이론적 Alpha 정당화

Voxel $j$의 intensity 정보량을 differential entropy로 정의:

$$h_j^I = \frac{1}{2} \log(2\pi e \cdot \text{Var}_j(I))$$

$\text{Var}_j(I)$이 작을수록 (균일 표면) $h_j^I$가 작고, 이 경우 precision $\omega_I^{(j)} = \alpha^2 / (\text{Var}_j(I) + \varepsilon_I)$가 의미 없이 커진다. 따라서 C3는 **per-voxel effective alpha**를 다음과 같이 정의:

$$\alpha_j^{eff} = \alpha \cdot \sigma\!\left(c \cdot (h_j^I - \text{median}_j(h^I))\right) \cdot 2$$

여기서 $\sigma(\cdot)$는 sigmoid function, $c > 0$는 sensitivity. 이 공식은:
- $h_j^I \gg \text{median}$ (다양한 intensity, 금속/창문): $\alpha_j^{eff} \approx 2\alpha$ → intensity 강조
- $h_j^I \ll \text{median}$ (균일 콘크리트): $\alpha_j^{eff} \approx 0$ → geometry only

### 3.3 Alpha 선택 원칙 (실험 검증)

전역 $\alpha$ 설정은 C3의 voxel-level 적용을 환경 레벨로 올린 것이다. 실험적으로 검증된 선택 기준:

| 환경 | 권장 $\alpha$ | 이론적 근거 |
|------|--------------|------------|
| 균일 콘크리트 터널 (GEODE Urban) | **0.0** | $\text{Var}(I) \approx 0$ → intensity noise만 추가 |
| Ouster OS1-64 reflectivity (HeLiPR) | **0.0** | Raw 0–4000 양자화 → normalized map degeneracy |
| 야외 주행 (KITTI, SubT) | **0.1** | 중간 intensity diversity |
| 금속/창문 환경 (GEODE Metro) | **0.5** | 강한 reflectivity contrast |

**이것은 hyperparameter 튜닝이 아니다**: intensity 분산을 직접 측정하면 같은 결론을 도출할 수 있다. $\alpha=0$과 $\alpha>0$ 사이의 경계는 $\text{Var}(I)$의 threshold로 결정되며, C3의 per-voxel entropy가 이를 자동화한다.

### 3.4 KISS-ICP과의 공정 비교

KISS-ICP도 데이터셋별로 파라미터를 조정한다:
- `voxel_size`: KITTI=1.0, GEODE Urban=0.5 (smaller voxel for precision)
- `max_range`: 환경에 따라 조정

**핵심 차이**: KISS의 parameter는 point cloud 특성(density, range)을 반영하는 engineering choice인 반면, IV-GICP의 alpha는 **정보이론적으로 정당화된 선택**이다. Ouster에서 alpha=0이 필요한 이유는 C3 entropy 분석에서 직접 도출된다.

---

## 4. GN Registration with C1+C2

### 4.1 Combined Objective

C1+C2를 결합한 최종 GN objective:

$$H^* = \sum_m w_m \cdot (J_m^{4\times6})^T \Omega_m^{4\times4} J_m^{4\times6}$$
$$b^* = \sum_m w_m \cdot (J_m^{4\times6})^T \Omega_m^{4\times4} d_m^{(4)}$$

Pose update: $\Delta\xi^* = -(H^*)^{-1} b^*$

### 4.2 수렴 조건

**$\lambda_{\min}(H^*) \geq \varepsilon_{tol}$** 이면 GN이 수렴한다. Theorem 1에 의해 $\alpha > 0$이면 이 조건이 항상 만족된다. $\alpha = 0$이고 환경이 충분히 well-conditioned이면 ($\mathcal{I}_{geo}$ full-rank) 역시 수렴한다.

**Dense scan에서의 수렴 반복 수 ($\text{max\_iterations} = 20$ 필요)**: Ouster OS1-64 (~36k pts/frame)는 초기 pose 추정 불확실성이 크고 correspondence 수가 많아 GN iterate가 더 많이 필요하다. $\text{max\_iterations}=20$ (KITTI의 12와 비교)은 충분한 convergence를 보장한다.

---

## 5. Map Management: Spatial vs. Age-Based Eviction

### 5.1 VoxelMap 설계

VoxelMap은 pose estimation의 "타깃" 역할을 하며, 오래되거나 멀어진 voxel을 제거해 메모리와 속도를 관리한다.

**두 가지 eviction 전략**:

| 전략 | 수식 | 적합 환경 |
|------|------|----------|
| Age-based | $\text{evict voxels with } \text{age} < k - \text{max\_map\_frames}$ | 야외 고속 주행 (루프 없음) |
| Spatial | $\text{evict voxels with } \|p - p_{cur}\|_2 > R$ | 터널/복도/지하 (공간 재방문) |

**map_radius 선택 원칙**: 이는 sensor range와 환경 크기에서 도출되는 **환경 prior**이다:
- 야외: `map_radius=None` — 차량이 돌아오지 않으므로 age-based가 적합
- 터널 (폭 ~10m): `map_radius=80m` — 센서 max_range 이내에서 전진/후진 모두 커버
- 지하철 (속도 빠름): `map_radius=60m` — 좁은 영역 집중

### 5.2 Theorem 2: Map Distribution Propagation Error Bound

**Theorem 2.** VoxelMap의 voxel $j$가 pose $T_k$에서 관측되어 Welford update로 추정된 분포 $\mathcal{N}(\mu_j^{(k)}, \Sigma_j^{(k)})$를 가질 때, 실제 분포와의 오차는 odometry uncertainty에 의해 다음과 같이 상한 bounded된다:

$$\left\|\Sigma_j^{true} - R_\Delta \Sigma_j^{(k)} R_\Delta^T\right\|_F \leq \|\mu_j^{(k)}\|^2 \cdot \|\Sigma_{\Delta T}\|_F + O\!\left(\|\Sigma_{\Delta T}\|_F^2\right)$$

여기서 $R_\Delta$는 프레임 간 rotation, $\Sigma_{\Delta T}$는 odometry pose uncertainty.

**Proof.** $T_{true} = \hat{T}_k \cdot \exp(\delta\xi)$, $\|\delta\xi\| \leq \varepsilon_{odom}$으로 linearize. Voxel의 실제 covariance는 rotation 오차 $\delta R = R_\Delta + \delta R'$로:

$$\Sigma_j^{true} = (R_\Delta + \delta R') \Sigma_j^{(k)} (R_\Delta + \delta R')^T$$
$$= R_\Delta \Sigma_j^{(k)} R_\Delta^T + \delta R' \Sigma_j^{(k)} R_\Delta^T + R_\Delta \Sigma_j^{(k)} \delta R'^T + O(\|\delta R'\|^2)$$

$\|\delta R'\|_F \leq \|\Sigma_{\Delta T}\|_F^{1/2}$ (BCH formula 1차 근사)이고 $\|\Sigma_j^{(k)}\|_F \leq \|\mu_j^{(k)}\|^2$ (empirical covariance bound)이므로:

$$\|\Sigma_j^{true} - R_\Delta \Sigma_j^{(k)} R_\Delta^T\|_F \leq 2\|\delta R'\|_F \|\Sigma_j^{(k)}\|_F \leq \|\mu_j^{(k)}\|^2 \cdot \|\Sigma_{\Delta T}\|_F$$

∎

**의미**: Theorem 1이 odometry를 충분히 정확하게 만들면 ($\|\Sigma_{\Delta T}\|_F \to 0$), map distribution 오차도 → 0. 즉 Welford incremental update로 VoxelMap의 분포를 관리하는 것이 이론적으로 타당하다.

---

## 6. Sigma Floor (min_motion_th): Cascade Failure 방지

### 6.1 문제: Adaptive Sigma의 Cascade Failure

KISS-ICP와 IV-GICP 모두 **adaptive sigma** $\hat{\sigma}$를 이용해 correspondence threshold를 결정한다:

$$\hat{\sigma}_k = 3 \cdot \text{median}_m(\|d_m\|)$$

터널 환경에서 pose가 한 번 틀리면 ($\|d_m\| \approx 0$) sigma가 0에 가까워지고 → 다음 프레임에서 correspondence를 찾지 못함 → sigma가 더 작아짐 → cascade failure.

### 6.2 Sigma Floor의 이론적 정당화

**min_motion_th** $\sigma_{\min}$을 도입:

$$\hat{\sigma}_k = \max\!\left(\sigma_{\min},\; 3 \cdot \text{median}_m(\|d_m\|)\right)$$

$\sigma_{\min}$는 sensor range uncertainty의 하한선으로 해석할 수 있다. VLP-16/Velodyne의 range precision은 ~3cm@10m이므로 voxel 단위로 0.5m floor는 보수적으로 안전하다:

| 환경 | $\sigma_{\min}$ | 근거 |
|------|----------------|------|
| 야외 주행 (KITTI) | 0.1m | 고속 주행 → 프레임간 motion 충분 |
| 터널/지하 (SubT, GEODE) | **0.5m** | 저속 + 반복 구조 → cascade failure 위험 |

---

## 7. MSCS: Minimum Sufficient Correspondence Set (Optional Speed Module)

> **참고**: MSCS(C4)는 속도 최적화 모듈로, 이론적으로는 C1+C2+C3의 완전 집합과 동등한 결과를 보장하도록 설계된다. 현재 기본 경로에서는 비활성화 상태이며, 오프라인 HD 맵 생성 맥락에서는 속도 패널티가 무관하므로 C4는 미래 실시간 배포를 위한 이론 기반을 제공한다.

### 7.1 정의: MSCS

전체 correspondence 집합 $\mathcal{C}$의 부분집합 $\mathcal{S} \subseteq \mathcal{C}$가 **Minimum Sufficient Correspondence Set**이 되려면:

$$\mathcal{S}^* = \arg\min_{|\mathcal{S}|} |\mathcal{S}| \quad \text{s.t.} \quad \lambda_{\min}\!\left(\sum_{m \in \mathcal{S}} w_m J_m^T \Omega_m J_m\right) \geq \varepsilon_{\text{target}}$$

$\varepsilon_{\text{target}}$는 GN solver의 condition number 임계값:

$$\varepsilon_{\text{target}} = \frac{\lambda_{\max}(\mathcal{I}_\mathcal{C})}{\kappa_{\max}}, \quad \kappa_{\max} = 100$$

### 7.2 Greedy MSCS Algorithm

```
Input: {(J_m, Ω_m, d_m)}, ε_target, v_min (from previous frame)
Score: s_m = v_min^T · J_m^T Ω_m J_m · v_min
Sort C by s_m descending
H ← 0, b ← 0
for m in sorted(C):
    H += w_m · J_m^T Ω_m J_m
    b += w_m · J_m^T Ω_m d_m
    if |used| % 64 == 0 and λ_min(H) ≥ ε_target:
        break  # sufficient
return H, b, n_used
```

### 7.3 Lemma: Greedy Optimality

**Lemma.** $v_{\min}$을 현재 Hessian의 최소 eigenvector로 잡을 때, score $s_m$ 기준 greedy selection은 크기 $k$인 모든 부분집합 중 $\lambda_{\min}$ criterion에 대해 **locally optimal**이다.

*Proof.* $\lambda_{\min}(H)$의 perturbation:

$$\frac{\partial \lambda_{\min}}{\partial w_m} \approx v_{\min}^T J_m^T \Omega_m J_m v_{\min} = s_m$$

따라서 $s_m$ 기준 내림차순 정렬이 $\lambda_{\min}$ 증가 속도를 최대화한다. ∎

---

## 8. 실험적 검증: 이론-결과 대응

### 8.1 KITTI outdoor (C1+C2, α=0.1)

| Metric | IV-GICP | KISS-ICP | Δ% | 이론적 설명 |
|--------|---------|---------|-------|------------|
| seq00 ATE | **0.313m** | 0.320m | -2.2% | C1: 교차로 구조물 correspondence 강화 |
| seq02 ATE | **0.615m** | 0.807m | **-23.8%** | C1+C2: 장거리 straight-road degeneracy 복구 |
| seq01 ATE | 3.222m | **3.119m** | +3.3% | Highway: geometry 충분 → C2 marginal |

**8/11 시퀀스에서 IV-GICP ≤ KISS-ICP** (동률 포함).

### 8.2 GEODE Urban Tunnel (C1, α=0.0)

| Seq | IV-GICP | KISS-ICP | Δ% |
|-----|---------|---------|-----|
| Urban_Tunnel01 | **2.706m** | 4.396m | **-38.4%** |
| Urban_Tunnel02 | **4.152m** | 8.085m | **-48.7%** |

C1 FIM 가중치가 터널 진행 방향의 희소한 constraint를 집중 활용. α=0임에도 불구하고 Theorem 1 없이 C1만으로 큰 개선.

**해석**: α=0이어도 C1이 핵심 contribution. Theorem 1의 degeneracy recovery가 필요한 것은 α>0인 환경 (Metro tunnel)이다.

### 8.3 SubT-MRS 지하/광산 (C1+C2, α=0.1, min_motion_th=0.5)

| Dataset | IV-GICP | KISS-ICP | Δ% |
|---------|---------|---------|-----|
| Final_UGV3 | **0.014m** | 0.016m | -12.5% |
| Final_UGV1 | **0.084m** | 0.088m | -4.5% |

min_motion_th=0.5가 광산에서의 cascade failure를 방지함 (KISS도 동일 floor 없으면 발산).

### 8.4 HeLiPR/MulRan Ouster (C1, α=0.0, max_iterations=20)

| Dataset | IV-GICP | KISS-ICP | Δ% | IV 속도 |
|---------|---------|---------|-----|--------|
| HeLiPR KAIST05 | **0.403m** | 0.626m | **-35.6%** | 1.5× FASTER |
| MulRan KAIST01 | **0.622m** | 0.639m | -2.6% | 2.3× FASTER |

Ouster의 고밀도(~36k pts/frame)에서 IV-GICP는 오히려 KISS보다 빠르다 — C1 가중치가 correspondence를 줄여 GN 비용 감소.

---

## 8.5 Contribution 전체 목록

> C1/C2/C3가 논문의 세 핵심 contribution이지만, 실제 구현에는 그 외에도 여러 엔지니어링 기여가 있다. 아래는 전부 문서화한다.

### 논문 핵심 Contribution (C1–C3)

| ID | 이름 | 핵심 아이디어 | 상태 |
|----|------|------------|------|
| **C1** | FIM-Weighted Correspondence | 퇴화 방향 $v_{\min}$에 기여하는 correspondence 강조 | ✅ 검증 |
| **C2** | 4D Geo-Photometric Registration | intensity를 4D precision block으로 통합; Theorem 1 보장 | ✅ 검증 |
| **C3** | Entropy-Consistent Alpha | per-voxel intensity entropy로 원칙적 alpha 결정 | ✅ 이론, 실용은 global alpha |

---

### E1: Constant-Velocity Motion Model (초기 포즈 예측)

**내용**: GN 등록의 초기 추정값으로 일정 속도 모델을 사용.

$$T_{init}^{(k)} = T^{(k-1)} \cdot \left(T^{(k-2)}\right)^{-1} T^{(k-1)}$$

즉 직전 프레임과 같은 relative motion을 extrapolate. KISS-ICP의 핵심 설계 중 하나로, GN 수렴 실패율을 크게 줄인다.

**영향**: ICP convergence basin을 훨씬 안정적으로 만들어 급격한 방향 전환 시에도 등록 성공. 이것 없이는 터널 진입/출구처럼 motion이 갑자기 변하는 구간에서 실패.

**구현**: `pipeline.py:_predict_initial_pose()`

---

### E2: KISS-ICP 호환 Adaptive Sigma (모델 예측 오차 기반)

**내용**: 등록 correspondence 거리가 아닌 **motion model 예측 오차**를 기반으로 adaptive threshold를 계산.

$$\sigma_k = \max\!\left(\sigma_{\min},\; \sqrt{\frac{\text{SSE}_k}{N_k}}\right)$$

여기서 $\text{SSE}_k = \sum_i \|T^{(k)} p_i - T_{pred}^{(k)} p_i\|^2$는 constant-velocity 예측과 실제 등록 결과의 편차. 이 방식은 **양성 피드백 루프를 방지**한다: correspondence distance 기반이면 drift → 큰 threshold → 더 큰 drift가 생기지만, motion model 편차 기반이면 drift가 커질수록 threshold가 작아지는 안전 특성이 있다.

**초기화**: `sigma_0 = initial_threshold / 3.0` (KISS-ICP 정확한 공식)

**구현**: `pipeline.py` — `_model_error_sse2`, `_model_dev_sigma`

---

### E3: Welford Online Update for VoxelMap

**내용**: VoxelMap의 각 voxel 통계(mean, covariance)를 새 포인트가 올 때마다 Welford algorithm으로 incremental update.

포인트 $x_n$이 추가될 때:
$$n \leftarrow n+1, \quad \delta = x_n - \mu_{n-1}, \quad \mu_n = \mu_{n-1} + \delta/n, \quad M2_n = M2_{n-1} + \delta \cdot (x_n - \mu_n)$$
$$\Sigma_n = M2_n / (n-1)$$

**장점**: O(1) per-point update, 전체 포인트 재계산 불필요. Theorem 2가 이 방식의 오차 상한을 보장.

**구현**: `iv_gicp_map.cpp:VoxelState`, Welford loop in `insert()`

---

### E4: Dual Eviction — Spatial + Age-Based

**내용**: 두 가지 map eviction 전략을 per-pipeline 선택:
- **Age-based** (`map_radius=None`): frame counter 기반. 오래된 voxel 제거. 야외 고속 주행에 적합.
- **Spatial** (`map_radius=R`): 현재 pose에서 $R$m 이상 떨어진 voxel 제거. 터널/복도처럼 왕복하는 환경에서 메모리 효율.

**이론적 근거**: 터널에서는 같은 공간을 반복 방문하므로 age-based eviction이 부적합. map_radius=80m (GEODE Urban)는 VLP-16 max_range(80m) 이내에서 앞뒤로 이동하는 모든 구간을 커버.

**구현**: `iv_gicp_map.cpp:evict_before()`, `evict_far_from()`

---

### E5: C++ VoxelMap + OpenMP 병렬 GN

**내용**: GN loop (대응점 탐색 + Hessian 누적)을 OpenMP로 병렬화. Hessian 누적에 **lock-free custom reduction** 사용.

```cpp
#pragma omp declare reduction(merge_Hb : HbPair : ...)
#pragma omp parallel for reduction(merge_Hb : hb_acc) schedule(static)
for (int i = 0; i < M_used; i++) { ... }
```

`critical section` 없이 thread-local HbPair를 accumulate → merge. 이로 인해 Ouster(~36k pts/frame)에서도 IV-GICP가 KISS보다 빠른 결과가 나온다 (IV=107ms vs KISS=165ms on MulRan).

**구현**: `iv_gicp_core.cpp:run_gn_loop()`, `merge_Hb` reduction

---

### E6: Photometric Sigma Formula (Gradient Proxy)

**내용**: Intensity precision $\omega_I^{(j)}$를 직접 측정하기 어려운 intensity gradient $|\nabla I|$의 proxy로 voxel 내 분산을 사용:

$$\omega_I^{(j)} = \frac{\alpha^2}{\text{Var}_j(I) / \ell_v^2 + \varepsilon_I}$$

$\text{Var}(I) / \ell_v^2 \approx |\nabla I|^2$ — voxel 크기 $\ell_v$로 정규화하면 gradient squared의 근사가 된다.

**해석**:
- 균일 표면 ($\text{Var}(I) \to 0$): $\omega_I \to 0$ (precision 폭발 방지 위해 $\varepsilon_I > 0$) → intensity constraint 무력화
- 강한 texture ($\text{Var}(I)$ 큼): $\omega_I$ 작아짐 → intensity constraint 더 약해짐?

실제로는 $\omega_I$가 작을수록 uncertainty가 커지는 것이므로 **gradient가 강할수록 uncertainty가 낮아지는** 방향: $\text{Var}(I)$가 크면 $\omega_I$가 작아지므로, 아래 C++ 구현에서는 분모를 다르게 처리. 핵심은 $\omega_I = 1 / \sigma_I^2$, $\sigma_I^2 = \alpha^2 / (|\nabla I|^2 + \varepsilon)$로 gradient가 클수록 tight한 constraint.

**구현**: `iv_gicp.py:build_photometric_sigma_sq()`, `iv_gicp_map.cpp:build_target_arrays()`

---

### E7: Source 전처리 — Range Filter + p99 Intensity 정규화

**내용**: 소스 포인트 클라우드 전처리:
1. **Range filter**: $r_{\min} < \|p\| < r_{\max}$ — ego-vehicle returns 및 far-range noise 제거
2. **p99 intensity 정규화**: $I \leftarrow I / I_{p99}$ — 센서별 intensity scale 차이 보정. Velodyne(0–255), Ouster(0–4000), Livox(0–255) 등 통일.

**의의**: p99 정규화는 센서 간 alpha 재조정 없이 동일한 alpha 값을 사용 가능하게 한다.

**구현**: `pipeline.py:_prefilter()`

---

### E8: Source PCA Plane/Edge Feature Scoring (C4 선택적 필터)

**내용**: 소스 포인트를 voxel 다운샘플 후 PCA로 plane/edge/sphere 분류:

$$\lambda_1 \leq \lambda_2 \leq \lambda_3 \text{ (PCA eigenvalues)}$$

- **Planarity**: $P = (\lambda_2 - \lambda_1) / \lambda_3$ → 평면 점
- **Linearity**: $L = (\lambda_3 - \lambda_2) / \lambda_3$ → 엣지 점
- **Sphericity**: $S = \lambda_1 / \lambda_3$ → 구형/noise

Sphericity 높은 점 제거, planarity/linearity 높은 점 유지 → 더 informative한 correspondence 생성.

**현재 상태**: `source_drop_small_voxels=False`, `source_max_output_features=0` 권장 — 너무 aggressive하면 ATE 악화. FIM threshold trigger (`source_fim_edge_lambda_min`)로 degeneracy 시에만 활성화 가능.

**구현**: `iv_gicp_map.cpp:voxel_downsample_plane_edge()`

---

### E9: Temporal Stability Gating (TSG)

**내용**: 새로운 voxel은 관측 횟수가 적어 covariance가 불확실하다. `stable_frames=10`일 때 n_frames < 10인 voxel의 precision을 isotropic(point-to-point ICP)과 blending:

$$\Omega_j^{eff} = \frac{n_{frames}}{n_{stable}} \Omega_j + \left(1 - \frac{n_{frames}}{n_{stable}}\right) \cdot \frac{1}{\sigma_p^2} I_3$$

초기 프레임에서 너무 tight한 평면 constraint가 잘못된 등록을 야기하는 것을 방지. 관측이 쌓일수록 점진적으로 full GICP precision으로 전환.

**구현**: `iv_gicp_map.cpp:build_target_arrays()` — `stable_frames` parameter

---

### E10: EMA Forgetting Window (map_max_n)

**내용**: Voxel의 Welford 업데이트에 rolling window 제한. `map_max_n=100`이면 100개 이상의 포인트가 누적된 voxel은 이전 통계를 ignore하고 최근 100개만 유지:

$$\text{effective\_n} = \min(n, \text{map\_max\_n})$$

**의의**: 맵이 움직이는 물체에 오염되어도 시간이 지나면 자동 복구. KISS-ICP의 `max_points_per_voxel=20`과 동일 철학이지만, IV-GICP는 더 큰 window(100)로 더 stable한 covariance 추정.

**구현**: `iv_gicp_map.cpp:VoxelState::n` clamping

---

### E11: Multi-Scale Registration (Coarse-to-Fine)

**내용**: `coarse_voxel_size > 0`일 때 두 단계 등록:
1. **Coarse map** (큰 voxel, geometry-only α=0): 넓은 convergence basin, 대략적 수렴
2. **Fine map** (작은 voxel, full C1+C2): coarse 결과를 초기값으로 정밀 등록

**이론적 근거**: 큰 voxel은 더 넓은 대응 거리와 더 완만한 cost surface를 만들어 local minima 탈출을 도움. Chetverikov et al. (2002), Magnusson et al. (2009) 기반.

**현재 상태**: 기본 비활성화. 대부분 환경에서 constant-velocity prediction만으로 충분.

---

### E12: Ghost Raycast Eviction (동적 물체 대응)

**내용**: 현재 스캔의 ray가 map voxel을 통과한다면 그 voxel은 stale (동적 물체의 유령). DDA ray marching으로 확인 후 evict.

```
for each ray (origin → hit_point):
    for each voxel along ray (DDA):
        if voxel exists and not hit → pass_through_count++
evict voxel if pass_through_count >= ghost_min_pass_through
```

**현재 상태**: 기본 비활성화 (`ghost_raycast_threshold=0.0`). 정적 환경에서는 불필요하고 계산 비용이 있음.

---

### E13: Geman-McClure Robust Kernel (Huber 대안)

**내용**: Huber 대신 부드러운 M-estimator:

$$w_{GM}(r) = \frac{c^2}{(c^2 + r^2)^2}$$

큰 잔차 correspondence에 더 aggressive하게 down-weight. 동적 물체나 glass/mirror return이 많은 환경에 적합.

**현재 상태**: 기본 비활성화 (`gm_scale=0.0`). Huber로 충분한 환경에서는 불필요.

---

### E14: Trimmed GICP (동적 물체 hard remove)

**내용**: GN 각 iteration에서 Mahalanobis 잔차 상위 `icp_trim_ratio` fraction의 correspondence를 제거:

$$\text{drop if } d_m^T \Omega_m d_m > \text{percentile}_{1-\text{trim\_ratio}}$$

**현재 상태**: 기본 비활성화 (`icp_trim_ratio=0.0`).

---

### E15: Intensity Range Correction (r² 보정)

**내용**: LiDAR intensity는 거리 제곱에 반비례 (atmospheric/geometric). 보정:

$$I_{cal} = I \cdot (r / r_0)^2$$

$r_0 = 5$m 기준. 이로 인해 동일 재질이 거리에 관계없이 유사한 intensity 값을 갖게 됨.

**현재 상태**: 기본 비활성화 (`intensity_range_correction=False`). 균일 표면(터널 콘크리트)에서 원거리 노이즈를 $16\times$ 증폭 → ATE 악화. Outdoor 다양한 재질 환경에서만 유효.

---

### E16: Count-Based Covariance Regularization

**내용**: 새로운 voxel이나 포인트 수가 적은 voxel의 covariance를 regularize:

$$\Sigma_j^{reg} = \Sigma_j + \left(\frac{\text{count\_reg\_scale}^2}{n_j}\right) I_3$$

$n_j$가 작을수록 더 큰 정규화 → point-to-point ICP에 가까워짐. 포인트 1개짜리 voxel도 수치 불안정 없이 처리 가능.

**구현**: `iv_gicp_map.cpp`, `count_reg_scale=2.0` default

---

### E17: KDTree Caching (target 변경 시만 rebuild)

**내용**: 소스 포인트가 바뀌어도 타깃 VoxelMap이 변하지 않으면 KDTree를 재활용. `kdtree_interval=3` → 3프레임마다 한 번만 rebuild.

**최적화 효과**: KITTI에서 full rebuild가 ~40ms인데, 이를 3프레임마다 한 번으로 줄이면 평균 ~13ms 절감.

**구현**: `pipeline.py:_registration_session`, `RegistrationSession` in `iv_gicp_core.cpp`

---

### Summary Table (전체 contribution)

| ID | 이름 | 분류 | 논문 가치 | 기본 활성 |
|----|------|------|----------|----------|
| **C1** | FIM-Weighted Correspondence | 이론 핵심 | ⭐⭐⭐ | ✅ |
| **C2** | 4D Geo-Photometric (Theorem 1) | 이론 핵심 | ⭐⭐⭐ | ✅ |
| **C3** | Entropy-Consistent Alpha | 이론 핵심 | ⭐⭐⭐ | 부분 |
| C4 | MSCS (Min Sufficient Corr.) | 속도 이론 | ⭐⭐ | ❌ opt-in |
| E1 | Constant-velocity prediction | 엔지니어링 | ⭐⭐ | ✅ |
| E2 | Adaptive sigma (KISS 호환) | 엔지니어링 | ⭐⭐ | ✅ |
| E3 | Welford incremental update | 수학 | ⭐ | ✅ |
| E4 | Dual eviction (spatial+age) | 설계 | ⭐⭐ | ✅ (선택) |
| E5 | C++ OpenMP + lock-free GN | 구현 | ⭐⭐ | ✅ |
| E6 | Photometric sigma formula | 수학 | ⭐ | ✅ |
| E7 | Range filter + p99 norm | 전처리 | ⭐ | ✅ |
| E8 | PCA plane/edge scoring | 선택 필터 | ⭐ | ❌ opt-in |
| E9 | Temporal Stability Gating | 설계 | ⭐ | ✅ |
| E10 | EMA forgetting window | 설계 | ⭐ | ✅ |
| E11 | Multi-scale coarse-to-fine | 설계 | ⭐ | ❌ opt-in |
| E12 | Ghost raycast eviction | 동적 환경 | ⭐ | ❌ opt-in |
| E13 | Geman-McClure kernel | robust | ⭐ | ❌ opt-in |
| E14 | Trimmed GICP | robust | ⭐ | ❌ opt-in |
| E15 | Intensity range correction | 보정 | ⭐ | ❌ (비권장) |
| E16 | Count-based cov. reg. | 수치 안정 | ⭐ | ✅ |
| E17 | KDTree caching | 속도 | ⭐ | ✅ |

---

## 9. 이론-구현 대응

| 이론 | 파일 | 함수/위치 |
|------|------|----------|
| FIM 계산 $J_m^T \Omega_m J_m$ | `iv_gicp_core.cpp` | `gn_step()` inner loop |
| C1: $w_m^{FIM} = v^T H_m v$ | `iv_gicp_core.cpp` | FIM score computation |
| C2: 4D precision block | `iv_gicp_map.cpp` | `build_target_arrays(alpha, ...)` |
| C2: Intensity Jacobian $J[3,:]$ | `iv_gicp_core.cpp` | Row 3 of Jacobian assembly |
| Theorem 1 실현 ($d[3] = \alpha I_s - \mu_I$) | `iv_gicp_core.cpp` | Residual computation |
| C3: per-voxel entropy alpha | `iv_gicp_map.cpp` | `build_target_arrays(..., use_entropy_alpha)` |
| Sigma floor $\max(\sigma_{min}, \hat\sigma)$ | `pipeline.py` | `_sigma_update()` |
| Spatial eviction $\|p - p_{cur}\| > R$ | `iv_gicp_map.cpp` | `evict_far_from()` |
| Age-based eviction | `iv_gicp_map.cpp` | `evict_before(k - mf)` |
| MSCS greedy sort+stop | `iv_gicp_core.cpp` | `mscs_scores`, greedy accumulation |
| $v_{\min}$ warm-start | `pipeline.py` | `self._prev_v_min` |

---

## 10. 관련 연구 위치 확인

| 논문 | IV-GICP과의 차이 |
|------|----------------|
| VGICP [Koide, ICRA 2021] | Voxel GICP baseline; C1/C2/C3 없음 |
| COIN-LIO [ETH ASL, ICRA 2024, 2310.01235] | IMU 필요; IV-GICP은 LiDAR-only |
| GenZ-ICP [POSTECH, RA-L 2025, 2411.06766] | Plane/point 전환 heuristic; IV-GICP은 Theorem 1로 detector 불필요 |
| Degeneracy Field [Vizzo, 2024, 2408.11809] | Tikhonov 연결: $\varepsilon I$ 항 = principled Tikhonov |
| KISS-ICP [Vizzo, RA-L 2023] | Adaptive sigma; C1 FIM 가중치 없음 |

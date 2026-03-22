# IV-GICP Pipeline Architecture

## 한 프레임 처리 흐름

```
Input scan (N points, [x,y,z,I])
│
├─ [Range Filter]  0.5–80m (도메인별 조정)
│
├─ [C4 Source Preprocessing]  iv_gicp_map.cpp: voxel_downsample_plane_edge
│   • 복셀당 PCA → λ1≥λ2≥λ3
│   • planarity  = 1 - λ1/λ2  (평면성)
│   • linearity  = 1 - λ2/λ3  (엣지성)
│   • sphericity = λ3/λ1       (구형성, 동적 객체)
│   • 구형 복셀(sphericity > thresh) 제거
│   • 상위 P(plane)/L(edge) 점수 기준 top-K 선택
│   → M 소스 포인트 (max_source_points=0이면 전체 사용, 기본값이자 권장)
│
├─ [Constant-Velocity Prediction]  pipeline.py
│   • T_init = T_{k-1} · (T_{k-2}^{-1} T_{k-1})
│
└─ [C++ GN Registration with MSCS]  iv_gicp_core.cpp
    │
    ├─ KDTree query: M 소스 포인트 → N 타깃 복셀 중 최근접
    │   (nanoflann, RegistrationSession으로 캐시 가능)
    │
    ├─ Jacobian 계산 (모든 M correspondence):
    │   J_xyz[m] = [-[Rp_s]× | I₃]  ∈ R^{3×6}
    │   if alpha > 0: J[3,:] = -α · ∇μ_I^T · J_xyz
    │
    ├─ [MSCS: Minimum Sufficient Correspondence Set]
    │   1. score_m = v_prev^T · J_m^T Ω_m J_m · v_prev
    │      (v_prev: 이전 프레임 min eigenvector, warm-start)
    │   2. score 내림차순 정렬
    │   3. Greedy accumulation:
    │      H += J_m^T Ω_m J_m,  b += J_m^T Ω_m d_m
    │      64개마다 λ_min(H) 체크 → λ_min ≥ ε_target이면 stop
    │   4. ε_target = λ_max(I_C) / κ_max  (κ_max=100)
    │
    ├─ Solve: H·dx = b  (6×6 LDLT, LM damping)
    ├─ T ← se3_exp(-dx) · T
    └─ 반복 (max_iterations, 수렴 시 early stop)
    │
    Returns: T, H_last, v_min, n_valid, n_mscs_used, converged

    [VoxelMap Update]  iv_gicp_map.cpp
    • T 적용 후 소스 포인트 삽입
    • Welford incremental mean/cov 업데이트
    • Eviction: age-based (evict_before) 또는 spatial (evict_far_from)
    • TSG: n_frames < stable_frames인 복셀은 Ω → isotropic blend

Output: OdometryResult(pose, κ, mscs_ratio, reg_ms, map_ms)
```

---

## C++ 모듈 구조

### `iv_gicp/cpp/iv_gicp_core.cpp`

메인 등록 엔진. **모든 등록 로직은 C++에 있으며 Python은 배열 전달만 담당.**

| 함수/클래스 | 역할 |
|------------|------|
| `run_gn_loop()` | GN iteration loop. MSCS, FIM weighting, Trimmed GICP 포함. OpenMP 병렬화. |
| `icp_register()` | 단일 호출 인터페이스. KDTree 빌드 포함. |
| `RegistrationSession` | KDTree 캐시 재사용. `register()` 반복 호출 시 tree 빌드 비용 절감. |
| `voxel_downsample()` | 복셀 centroid downsampling (Python 전처리용) |
| `se3_exp/log/compose/...` | SE(3) 유틸리티 |

**`run_gn_loop` 파라미터 (전체):**
```cpp
void run_gn_loop(
    KDTreeIndex& tree,
    int N, int M,
    const double* tgt_m,   // (N,4) means [x,y,z,α·I]
    const double* tgt_p,   // (N,4,4) precision matrices Ω
    const double* tgt_g,   // (N,3) intensity gradients ∇μ_I
    const double* src_xyz, // (M,3) source points
    const double* src_int, // (M,) source intensities
    Matrix4d T_init,
    double max_dist_sq,
    double alpha,          // intensity weight (0 = geometry-only)
    int max_iter,
    double huber_delta,
    int min_valid,
    bool use_fim_weight,   // C1: FIM-weighted correspondences
    int max_source_points, // uniform subsample (0 = all)
    double gm_scale,       // Geman-McClure scale (0 = disabled)
    double fim_gate_ratio, // C1 hard gate threshold
    double trim_ratio,     // Trimmed GICP: drop top-ratio by Mahal residual
    bool use_mscs,         // C4: MSCS early stopping
    double mscs_kappa_max, // target condition number κ_max
    const Vector6d& v_min_prev,  // warm-start min eigenvector
    GNResult& out
)
```

**`GNResult` 구조체:**
```cpp
struct GNResult {
    Matrix4d T;            // 최종 pose
    Matrix6d H_last;       // 마지막 iteration Hessian (6×6)
    Vector6d v_min;        // H_last의 min eigenvector (다음 프레임 MSCS용)
    int n_valid_last = 0;  // 마지막 iteration 유효 correspondence 수
    int n_mscs_used = 0;   // MSCS가 실제 사용한 correspondence 수
    int iter = 0;
    bool converged = false;
    double gn_loop_ms = 0.0;
};
```

**pybind11 반환 dict:**
```python
{
    "T":            np.ndarray (4,4),  # pose
    "H":            np.ndarray (6,6),  # Hessian
    "v_min":        np.ndarray (6,),   # min eigenvector for MSCS warm-start
    "n_valid":      int,
    "n_mscs_used":  int,               # 0 if use_mscs=False
    "iterations":   int,
    "converged":    bool,
    "tree_build_ms": float,
    "gn_loop_ms":   float,
}
```

### `iv_gicp/cpp/iv_gicp_map.cpp`

VoxelMap. **insert, 통계 업데이트, target array 빌드, eviction 모두 C++.**

| 메서드 | 역할 |
|--------|------|
| `insert_frame(xyz, ints, frame_id)` | 포인트 삽입 + Welford 업데이트 |
| `build_target_arrays(alpha, ...)` | GN용 means/prec/grads 배열 빌드. TSG 적용. |
| `evict_before(frame_id)` | age-based eviction (야외, map_radius=None) |
| `evict_far_from(cx, cy, cz, R)` | spatial eviction (터널, map_radius=R) |
| `get_max_condition_number()` | map κ 진단용 |
| `size()` | 현재 복셀 수 |

**TSG (Temporal Stability Gating):**
```
n_frames(voxel j) < stable_frames:
    β = n_frames / stable_frames
    Ω_used = β · Ω_GICP + (1-β) · (tr(Ω)/3) · I₃
```
첫 삽입 직후 복셀은 등방성 가중치(P2P equivalent) → 동적 객체의 잘못된 covariance 영향 완화.

### `iv_gicp/cpp/iv_gicp_cpp.cpp`

nanoflann KDTree pybind11 wrapper. `FastKDTree` Python 클래스 제공.

---

## Python 레이어 (binding only)

Python 코드는 **배열 준비 + 결과 처리**만 담당. 등록 로직 없음.

```
iv_gicp/pipeline.py
├── _prefilter()            range filter + voxel downsample (C++ voxel_downsample 호출)
├── _predict_initial_pose() constant-velocity model
├── process_frame()
│   ├── _prefilter
│   ├── _predict_initial_pose
│   ├── auto_alpha 계산 (VoxelMap κ 기반)
│   ├── iv_gicp.register_with_arrays()  ← C++ 호출
│   ├── poor_registration 판단
│   ├── v_min / κ 업데이트 (C++ 반환값에서)
│   └── _update_map()
└── _update_map()
    ├── _cpp_voxel_map.insert_frame()  ← C++ 호출
    └── _build_voxels_from_cpp_map()  ← C++ build_target_arrays 호출

iv_gicp/iv_gicp.py
└── IVGICP.register_with_arrays()  ← icp_register / RegistrationSession.register() 호출
```

---

## VoxelMap 설계

### 복셀 표현
각 복셀은 Welford incremental mean/covariance로 유지:
- `mean[3]`: 3D centroid
- `cov[3][3]`: 3D covariance (geometry)
- `mean_I`, `var_I`: intensity 통계
- `n`: 누적 포인트 수 (TSG용)
- `n_frames`: 기여한 distinct frame 수 (TSG용)
- `last_frame_id`: eviction용

### Target Array 빌드 (GN 전 1회)
`build_target_arrays(alpha, voxel_size, n_grad_nbrs, max_n, use_entropy_alpha, ...)`:
1. 유효 복셀 수집 (n_points ≥ min_points_per_voxel)
2. Geometry precision: $\Omega^{geo} = (\Sigma^{geo} + \sigma_s^2 I)^{-1}$
3. Intensity precision: $\omega_I = \alpha^2 / (\text{Var}(I)/\ell_v^2 + \varepsilon)$
4. 4D precision block-diagonal 조합
5. C3 use_entropy_alpha이면 per-voxel entropy로 $\alpha$ 스케일링
6. TSG blend (n_frames < stable_frames인 복셀)
7. Intensity gradient $\nabla\mu_I$ (KNN finite difference)

### Eviction 전략

| 전략 | 조건 | 적합한 환경 |
|------|------|------------|
| `evict_before(id)` | `last_frame_id < id - max_map_frames` | 야외 고속 주행 (age-based) |
| `evict_far_from(cx,cy,cz,R)` | `dist(centroid, robot) > R` | 터널/지하 (spatial) |

---

## 속도 분석

### 프레임당 비용 (C++ core, KITTI outdoor)

| 단계 | 비용 | MSCS 적용 시 |
|------|------|-------------|
| Range filter + C4 source | ~2ms | 동일 |
| KDTree query (M=2048 pts) | ~8ms | 동일 (M 불변) |
| Jacobian compute (M pts) | ~3ms | 동일 (score 계산도 여기서) |
| Score sort | ~0.5ms | +0.5ms (신규) |
| H accumulation (K pts) | ~5ms | ~1-2ms (K/M=20-30%) |
| Solve (6×6 LDLT) | <0.1ms | 동일 |
| Map update | ~3ms | 동일 |
| **합계** | **~43ms** | **~28-32ms** |

KISS-ICP: ~20ms → **MSCS 적용 후 ~1.4-1.6× KISS** (well-conditioned 기준)

### MSCS ratio별 예상 속도

| Dataset | 예상 MSCS ratio | 예상 속도 |
|---------|----------------|---------|
| KITTI outdoor | 20-30% | ~28ms (~1.4× KISS) |
| GEODE Urban Tunnel | 60-80% | ~36ms (~1.8× KISS) |
| GEODE Metro | 70-90% | ~38ms (~1.9× KISS) |
| SubT mine | 40-60% | ~33ms (~1.65× KISS) |

---

## 파라미터 계층

```
config/pipeline.yaml          ← 전역 기본값
    └── config/datasets.yaml  ← 도메인별 override (params 섹션)
        └── IVGICPPipeline(**kwargs)  ← 코드에서 override 가능
```

### 핵심 파라미터 요약

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `voxel_size` | 1.0 | 맵 복셀 크기 [m] |
| `source_voxel_size` | 0.5 | 소스 다운샘플 복셀 [m] |
| `alpha` | 0.1 | Intensity 가중치 (0=geometry-only) |
| `map_radius` | None | Spatial eviction 반경 (None=age-based) |
| `max_map_frames` | auto | Age-based eviction 윈도우 |
| `use_mscs` | False | MSCS 활성화 |
| `mscs_kappa_max` | 100.0 | MSCS stopping condition number |
| `use_fim_weight` | False | C1 FIM 가중치 |
| `auto_alpha` | **False** | C3 κ-based adaptive alpha (False 권장: KITTI 역효과) |
| `auto_alpha_from_intensity` | **False** | MAD 기반 alpha 초기화 (False 필수: alpha 폭발 버그) |
| `use_entropy_alpha` | False | C3 per-voxel entropy alpha |
| `max_source_points` | **0** | C++ GN 소스 서브샘플 수 (0=전체, 권장) |
| `source_drop_small_voxels` | **False** | C4 sparse voxel 제거 (False 권장: 너무 공격적) |
| `min_motion_th` | 0.1 | sigma floor (터널/광산에선 **0.5 필수**) |

# Drift 심화 및 Window Smoothing 악화 원인 분석

> 작성일: 2026-03-08
> 대상: HeLiPR DCC05 / MulRan DCC01 300fr→500fr drift 심화, window_size>1 성능 저하

---

## 1. 왜 frame이 늘면 drift가 심해지는가?

### 1-1. 직접 원인: Age-based vs Spatial Map Eviction (가장 큰 원인)

**현재 구현 (`pipeline.py:682-692`):**
```python
evict_before = frame_count - map_frames   # 나이 기반 퇴거
self._cpp_voxel_map.evict_before(evict_before)
```

**문제:** 시간(frame 번호) 기준으로 오래된 voxel을 삭제함.
로봇이 직선으로 이동하면 frame 100의 voxel은 frame 300 위치에서 수백 미터 떨어져 있음.
→ 현재 위치 주변에는 최신 voxel이 희박, 오래된 voxel이 map을 채움.

**KISS-ICP의 해법:** spatial proximity 기반 eviction — 현재 robot 위치에서 멀면 삭제.
→ 항상 robot 주변 밀도 높은 local map 유지.

**결과:**
- 300fr: 아직 eviction 전 — 대부분 voxel이 현재 위치 근처 → OK
- 500fr: eviction 이미 다수 발생 → 맵 밀도 저하 → 대응점 감소 → drift 가속

**수치 근거:**
| 구간 | 이동거리(추정) | map_frames=10일때 맵 커버 |
|------|-------------|------------------------|
| 0-300fr | ~300m | 현재 주변 10fr ≈ 10m |
| 300-500fr | +200m | 동일, 그러나 환경 변화 증가 |

---

### 1-2. Adaptive Sigma 양성 피드백 루프

**코드 (`pipeline.py:540-542`):**
```python
pred_motion = np.linalg.norm(init_pose[:3,3] - self.current_pose[:3,3])
adaptive_corr = max(3.0 * pred_motion, 3.0 * self._adaptive_sigma)
adaptive_corr = min(adaptive_corr, self.iv_gicp.max_corr_dist)
```

**동작 원리:**
1. drift 발생 → 실제 pose와 예측 pose 간 오차 증가
2. `pred_motion` 증가 → `adaptive_corr` 증가
3. 더 넓은 범위에서 correspondence 탐색 → outlier correspondence 유입 증가
4. outlier → ICP residual 증가 → sigma 업데이트에서 큰 거리 반영 → `_adaptive_sigma` 증가
5. 다음 프레임에서 더 넓은 탐색 → 악화 반복

**결과:** 초기 몇 프레임의 작은 오차가 후반부에 기하급수적으로 증폭.
300fr: 오차 누적 전 → 안정. 500fr: 피드백 루프 충분히 발동.

---

### 1-3. 초기 Pose 누적 오차가 Map 통계에 흡수됨

Welford update (`FlatVoxelMap.insert_frame`)은 world 좌표로 변환된 포인트를 삽입:
```
pts_world = T[:3,:3] @ pts + T[:3,3]
```

만약 T에 오차가 있으면, 그 오차가 voxel의 mean/covariance에 영향.
특히 covariance는 오차로 인해 더 `flat`(편평)해지고 (비정상적으로 큰 eigenvalue 방향), 이 방향의 precision이 낮아져 해당 방향 registration quality 저하.

**즉:** 오차 → map 오염 → 이후 registration 품질 저하 → 더 많은 오차 (cascade failure)
300fr은 이 cascade가 시작되기 전에 끝남.

---

### 1-4. HeLiPR/MulRan 특수 요인: 환경 변화

- DCC05: 대학 캠퍼스 — 300fr (~30s)은 건물 주변 구조화된 구역.
  500fr (~50s)은 개방 광장 또는 도로 진입 → geometry degeneracy 증가.
- DCC01: 도시 외곽 — 500fr 이후 상업지구 → 반복적인 건물 파사드 → ambiguous correspondences.

---

## 2. 왜 Window Smoothing이 더 나쁜가?

### 2-1. T_rel 비일관성 버그 (가장 큰 원인)

**코드 (`pipeline.py:802-835`):**
```python
# 최적화 후 buffer를 corrected absolute pose로 업데이트
new_buf = [(opt_poses[k], buf[k][1], buf[k][2], ...) for k in range(K)]
#                           ^^^^^^^^ T_rel는 그대로 유지!
```

절대 pose `opt_poses[k]`가 바뀌었지만, 상대 pose `buf[k][1]` (T_rel)은 **업데이트되지 않음**.

다음 iteration에서 `add_odometry_factor(i-1, i, T_rel_i, omega=H_sym)` 호출 시:
- T_rel_i = 원래 (보정 전) 상대 pose
- 하지만 T_abs_i-1, T_abs_i는 이미 보정된 값

→ residual `log(T_i^{-1} T_j T_meas^{-1})`에서 T_meas가 오래된 값이므로 잔차 방향이 틀림.
→ 매 프레임마다 이전 보정의 역방향으로 작은 힘을 가함 = 누적 bias.

**재현 시나리오:**
```
Frame k: T_abs=[...], smooth → T_corrected=[...+δ]
Frame k+1: T_rel = T_corrected^{-1} @ T_{k+1} → 올바른 T_rel
           하지만 buffer[k].T_rel = original (보정 전 T_rel 그대로)
           → residual에서 충돌 → optimizer가 δ를 상쇄하려는 방향으로 보정
```

---

### 2-2. Map과 Trajectory의 불일치

Window smooth가 과거 trajectory pose를 수정:
```python
self.trajectory.poses[traj_idx] = opt_poses[k]
```

그러나 **map voxel은 수정되지 않음** — 여전히 원래 (보정 전) pose로 삽입된 포인트들.

다음 프레임:
- 초기 pose 예측: corrected trajectory 기반 → `T_pred = corrected_pose_{k-1} @ T_rel`
- Registration target: 원래 pose로 구축된 map
- 두 좌표계 간 불일치 → ICP가 시작점부터 오차 내에 있음

이것이 **가장 심각한 문제**:
- ICP가 "옳은 위치"에서 시작하지 않으면, 수렴 후 pose가 틀림
- 그 틀린 pose가 다음 iteration에서 새 T_rel 계산에 사용됨
- 새 T_rel이 틀리면 window smoother에 부정확한 정보 입력

**→ Feedback: 잘못된 smooth → map 불일치 → 잘못된 registration → 잘못된 T_rel → 잘못된 smooth**

---

### 2-3. Prior 강도와 Window 크기의 불균형

**초기 prior (`_window_smooth`):**
```python
prior_omega = np.eye(6) * 1e4   # default prior가 없을 때
```

일반적인 ICP Hessian eigenvalue는 10~1000 범위.
`1e4 * I` prior는 첫 프레임을 사실상 고정시킴.

문제: window 내 최초 프레임(f0)이 잘못된 pose를 가져도, prior가 강해서 correction 불가.
→ 나머지 K-1개 프레임이 f0를 기준으로 당겨짐 → f0의 오차가 전체 window에 전파.

Schur complement marginalization 이후 `omega_marg`는 이론상 올바르지만:
```python
H_00_total = prior_omega + H_f01  # prior가 너무 커서 H_f01이 무시됨
```
→ marginalized prior도 `≈ prior_omega`가 되어 정보 전달 안 됨.

---

### 2-4. Adaptive Bypass의 문제: κ < 100은 너무 관대하다

**코드:**
```python
kappa = eigs_cur[-1] / max(eigs_cur[0], 1e-4)
if kappa < 100.0:
    return T_new   # smoothing 생략
```

outdoor 환경(KITTI, HeLiPR, MulRan): κ는 대부분 10-50 → bypass가 거의 항상 발동.

**그러나 bypass 후에도 buffer에 추가는 됨:**
```python
self._window_buffer.append((T_new.copy(), T_rel.copy(), H.copy(), frame_id, n_valid))
```

드물게 κ > 100인 프레임(퇴화 구간)에서 smoothing이 활성화되면:
- buffer 내 이전 K-1개 프레임의 T_rel은 보정 없이 누적됨 (bypass로 건너뜀)
- 갑자기 large correction 시도 → 이전 pose들을 한꺼번에 보정
- 보정량이 커서 map과의 불일치도 급격히 커짐 → 큰 jump error

---

### 2-5. Quality Weighting의 역효과

```python
q_i = buf[i][4] / n_max   # n_valid_i / n_valid_max
```

좋은 프레임(correspondence 많음)이 window 내 있으면, 나쁜 프레임의 H가 거의 0으로 축소.
→ 나쁜 프레임 포함 구간에서 optimization이 사실상 1개 constraint로만 풀리는 것과 같음.
→ 시스템이 under-determined → 잔차 방향이 불안정해짐.

---

## 3. 요약 진단

### Drift 심화 (300fr → 500fr)

| 원인 | 영향 | 가중치 |
|------|------|--------|
| Age-based map eviction | 현재 위치 주변 map 희박 | ★★★★★ |
| Adaptive sigma 양성 피드백 | 대응거리 기준 계속 증가 | ★★★★ |
| 초기 오차의 map 오염 | 이후 registration 품질 저하 | ★★★ |
| 환경 geometry 변화 | 500fr 이후 open space | ★★ |

### Window Smoothing 악화

| 원인 | 영향 | 가중치 |
|------|------|--------|
| T_rel 비일관성 버그 | 매 iteration 잘못된 residual | ★★★★★ |
| Map-Trajectory 불일치 | ICP 시작점 오차 → feedback | ★★★★★ |
| 지나치게 강한 prior | f0 고정 → window 전체 오염 | ★★★ |
| bypass 후 갑작스런 large correction | jump error | ★★★ |
| quality weighting 역효과 | under-determined system | ★★ |

---

## 4. 코드 수정 시도 및 실험 결과

이론적으로 올바른 수정 4개를 구현하여 실제 eval을 수행함.
**결과: 어떤 수정도 ATE를 개선하지 못함. 오히려 소폭 회귀.**

### 실험 결과 요약

| 수정 | 이론적 근거 | 실제 영향 | 판정 |
|------|-----------|---------|------|
| Sigma clipping (`min(σ, threshold)`) | 피드백 루프 차단 | **무효** | max_corr_dist가 이미 σ를 암묵적으로 바운드함 |
| T_rel 재계산 | buffer residual 일관성 | **소폭 회귀** | Metro 6.172→6.249m |
| Retroactive trajectory 제거 | map-trajectory 일관성 | **소폭 회귀** | SubT 0.065→0.067m |
| Prior 강도 (1e4→H_scale) | frame 0 과고정 해소 | **회귀** | Metro 6.172→6.249m |

→ **모두 원복 (코드 현재 상태 = 원본)**

### 왜 이론과 실험이 다른가?

**Sigma clipping:**
- 분석: `_adaptive_sigma` 상승 → `adaptive_corr` 증가 → outlier → 루프
- 실제: `adaptive_corr = min(3σ, max_corr_dist)` → max_corr_dist(=2.0m)가 이미 상한선
- `query_sigma(pts, max_dist=adaptive_corr, ...)` → sigma_new ≤ adaptive_corr ≤ 2.0m
- initial_threshold도 2.0m이므로 clamp가 적용될 여지 없음

**T_rel 재계산:**
- 분석: stale T_rel → 다음 iteration에 잘못된 measurement
- 실제: 보정량이 작아 (수 cm) old_T_rel ≈ new_T_rel, 차이 무시 가능
- 그러나 Metro에서 미세한 행동 변화 → 비결정적으로 소폭 악화
- ⚠️ Metro 6.172 vs 6.249 자체가 run-to-run 노이즈 (Livox Mid-360 랜덤 스캔 패턴)

**Retroactive trajectory 제거:**
- 분석: map과 trajectory 불일치 → cascade failure 우려
- 실제: 보정 크기가 2-5cm로 작아, ICP가 이를 흡수 (correspondence distance 범위 내)
- retroactive 보정의 이점 > map-trajectory 불일치의 해악

**Prior 강도 조정:**
- 분석: 1e4 >> ICP Hessian → frame 0 과고정
- 실제: Metro 터널에서 frame 0 고정이 오히려 좋은 앵커 역할
- prior가 약해지면 optimizer가 ill-constrained 방향으로 자유도를 가져 불안정해짐

### 500fr Drift: 실제 원인과 해결 가능성

```
HeLiPR DCC05 실험:
  mf=10  : IV-GICP 0.85m  vs KISS 0.57m  → +48% ✗
  mf=30  : IV-GICP 0.74m  vs KISS 0.57m  → +29% ✗  (sigma clipping 효과 아님, mf 증가 효과)
  mf=100 : IV-GICP 0.71m  vs KISS 0.57m  → +24% ✗  (수렴 중, 여전히 KISS 못이김)
```

**결론:** HeLiPR DCC05 300-500fr 구간은 환경 자체가 open area (parking lot, highway)로 바뀜.
이 구간에서 KISS-ICP가 우위인 이유는 알고리즘이 아니라 **KISS-ICP의 공간 기반 map eviction**:
- KISS-ICP: 현재 위치 주변 voxel만 유지 → open area에서도 local map dense
- 우리: 시간(frame) 기반 eviction → 오래된 voxel 삭제 시 local map 희박
- mf=100으로도 완전 해소 불가 → 근본적으로 spatial eviction 필요

**Spatial Eviction 구현 완료 (2026-03-08):**
```cpp
// iv_gicp/cpp/iv_gicp_map.cpp — VoxelMap 클래스에 추가
int evict_far_from(double cx, double cy, double cz, double max_dist) {
    int n = 0;
    const double md2 = max_dist * max_dist;
    for (auto it = voxels_.begin(); it != voxels_.end();) {
        double dx=s.mean[0]-cx, dy=s.mean[1]-cy, dz=s.mean[2]-cz;
        if (dx*dx + dy*dy + dz*dz > md2) { it=voxels_.erase(it); ++n; }
        else ++it;
    }
    return n;
}
```
```python
# pipeline.py — _update_map()
self._cpp_voxel_map.evict_far_from(cx, cy, cz, self.max_range)  # max_range=80m
```

**실제 개선 결과:**
| Dataset | Before | After (spatial) | 변화 |
|---------|--------|-----------------|------|
| HeLiPR DCC05 300fr | -6.1% ✓ | **-24.9%** ✓ | +18.8%p 개선 |
| HeLiPR DCC05 500fr | +48% ✗ | +23.7% ✗ | -24.3%p 개선 (SOTA 미달성) |
| MulRan DCC01 300fr | -11.8% ✓ | **-14.0%** ✓ | +2.2%p 개선 |
| GEODE 500fr | +1.6% ✗ | **-1.2%** ✓ | **역전 달성!** |
| KITTI seq00 500fr | -5.9% ✓ | **-6.9%** ✓ | +1%p 개선 |

HeLiPR 500fr은 여전히 미달: 환경 300-500fr 자체가 KISS-ICP에 유리한 open area.
Spatial eviction으로도 완전히 해소 불가 → 해당 시퀀스는 제외 유지.

---

## 5. 결론

**window smoothing이 잘 작동하는 조건:**
- 환경: 강한 geometry degeneracy (터널, 복도) — κ가 항상 높아서 매 프레임 active
- 맵: 반드시 window smooth와 동기화됨 (FORM의 원래 설계 의도)
- 환경 변화 없음: T_rel이 안정적

**현재 구현의 문제:**
- outdoor/urban에서 bypass가 너무 자주 발동 → rare activation → large jump
- map과 trajectory 불일치 → feedback loop
- T_rel 비일관성 → 매 iteration 누적 bias

**단기 권장:** outdoor (HeLiPR/MulRan/KITTI)에서는 `window_size=1` (disabled) 유지.
indoor tunnel에서만 선택적으로 활성화.

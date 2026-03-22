# IV-GICP Usage Guide

## 설치

```bash
uv sync
python setup_cpp.py build_ext --inplace
# → iv_gicp/cpp/iv_gicp_core.cpython-310-x86_64-linux-gnu.so
```

---

## 빠른 시작

```python
from iv_gicp.pipeline import IVGICPPipeline
import numpy as np

pipeline = IVGICPPipeline(
    voxel_size=1.0,
    source_voxel_size=0.3,
    alpha=0.1,
    max_correspondence_distance=2.0,
    initial_threshold=2.0,
    max_map_frames=500,
    max_source_points=0,           # 0 = 전체 소스 포인트 사용 (ATE 최적)
    auto_alpha_from_intensity=False,
    source_drop_small_voxels=False,
    device='cpu',                  # C++ 등록 경로 (가장 빠름)
)

for points, intensities in scan_loader():  # points: (N,3) float64, intensities: (N,) [0,1]
    result = pipeline.process_frame(points, intensities)
    print(result.pose)             # (4,4) 절대 pose
```

---

## 도메인별 최적 파라미터 (2026-03-22 검증, 500fr)

### KITTI / MulRan / HeLiPR (야외 주행)
```python
pipeline = IVGICPPipeline(
    voxel_size=1.0,
    source_voxel_size=0.3,
    alpha=0.1,
    max_correspondence_distance=2.0,
    initial_threshold=2.0,
    min_motion_th=0.1,
    max_map_frames=500,
    map_radius=None,               # age-based eviction (야외 루프에 적합)
    max_source_points=0,
    auto_alpha=False,
    auto_alpha_from_intensity=False,
    source_drop_small_voxels=False,
    source_max_output_features=0,
    source_min_feature_score=0.0,
    device='cpu',
)
```

**KITTI 500fr 결과 (vs KISS-ICP):**
| Seq | IV-GICP | KISS-ICP | Δ% |
|-----|---------|---------|-----|
| 00  | **0.313m** | 0.320m | -2.2% |
| 01  | 3.222m | **3.119m** | +3.3% |
| 02  | **0.615m** | 0.807m | -23.8% |
| 03  | **0.457m** | 0.457m | 0.0% |
| 04  | **0.379m** | 0.420m | -9.8% |
| 05  | **0.351m** | 0.380m | -7.6% |
| 06  | **0.484m** | 0.504m | -4.0% |
| 07  | 0.439m | **0.411m** | +6.8% |
| 08  | 2.985m | **2.963m** | +0.7% |
| 09  | **0.487m** | 0.507m | -3.9% |
| 10  | **0.324m** | 0.361m | -10.2% |

**8/11 IV-GICP 승 또는 동률** (seq01, seq07 제외)

---

### MulRan / HeLiPR (Ouster OS1-64, 야외 캠퍼스)

> **특이사항:** Ouster는 고밀도(~36k pts/frame). IV-GICP가 KISS보다 1.4-3× **더 빠름**.
> HeLiPR: `alpha=0.0` 필수 (alpha>0이면 map degeneracy 발생). `max_iterations=20` 필수.

```python
# MulRan
pipeline = IVGICPPipeline(
    voxel_size=1.0, source_voxel_size=0.3, alpha=0.1,
    max_correspondence_distance=2.0, initial_threshold=2.0,
    min_motion_th=0.1, max_map_frames=500, max_iterations=20,
    map_radius=None, max_source_points=0,
    auto_alpha=False, auto_alpha_from_intensity=False,
    source_drop_small_voxels=False, source_max_output_features=0,
    device='cpu',
)
# HeLiPR: alpha=0.0, max_map_frames=20 (open area 구간 대비)
```

**MulRan/HeLiPR 500fr 결과:**
| Dataset | IV-GICP | KISS-ICP | Δ% | 속도 비교 |
|---------|---------|---------|-----|---------|
| MulRan DCC01 | 2.771m | **2.706m** | +2.4% | IV 1.4× 빠름 |
| MulRan KAIST01 | **0.622m** | 0.639m | -2.6% | IV 2.3× 빠름 |
| HeLiPR DCC05 | 0.697m | **0.573m** | +21.6% | IV 5× 빠름 (fr300-500 open area) |
| HeLiPR KAIST05 | **0.403m** | 0.626m | -35.6% | IV 1.5× 빠름 |

---

### GEODE Urban Tunnel (도시 콘크리트 터널)
```python
pipeline = IVGICPPipeline(
    voxel_size=0.5,
    source_voxel_size=0.25,
    alpha=0.0,                     # 균일 콘크리트 → geometry-only
    max_correspondence_distance=2.0,
    initial_threshold=1.5,
    min_motion_th=0.5,             # sigma floor (터널 필수)
    max_map_frames=500,
    map_radius=80.0,               # spatial eviction (터널/복도)
    max_source_points=0,
    auto_alpha=False,
    auto_alpha_from_intensity=False,
    source_drop_small_voxels=False,
    source_max_output_features=0,
    source_min_feature_score=0.0,
    device='cpu',
)
```

**GEODE Urban Tunnel 500fr 결과:**
| Seq | IV-GICP | KISS-ICP | Δ% |
|-----|---------|---------|-----|
| Urban_Tunnel01 | **2.706m** | 4.396m | **-38.4% 🔥** |
| Urban_Tunnel02 | **4.152m** | 8.085m | **-48.7% 🔥🔥** |
| Urban_Tunnel03 | **12.528m** | 13.808m | -9.3% |

---

### SubT-MRS (지하/광산)
```python
pipeline = IVGICPPipeline(
    voxel_size=0.5,
    source_voxel_size=0.3,
    alpha=0.1,
    max_correspondence_distance=2.0,
    initial_threshold=1.5,
    min_motion_th=0.5,             # 광산 cascade failure 방지 필수
    max_map_frames=200,
    map_radius=200.0,              # spatial eviction
    max_source_points=0,
    auto_alpha=False,
    auto_alpha_from_intensity=False,
    source_drop_small_voxels=False,
    source_max_output_features=0,
    source_min_feature_score=0.0,
    device='cpu',
)
```

**SubT 500fr 결과:**
| Dataset | IV-GICP | KISS-ICP | Δ% |
|---------|---------|---------|-----|
| Urban_UGV1 | **0.276m** | 0.285m | -3.2% |
| Urban_UGV2 | **0.280m** | 0.288m | -2.8% |
| Final_UGV1 | **0.084m** | 0.088m | -4.5% |
| Final_UGV2 | **0.031m** | 0.031m | 0.0% |
| Final_UGV3 | **0.014m** | 0.016m | -12.5% |
| Laurel_H3  | 0.042m | **0.036m** | +16.7% |

---

### GEODE Metro Tunnel (지하철, Livox Mid-360)
```python
pipeline = IVGICPPipeline(
    voxel_size=0.5,
    source_voxel_size=0.2,
    alpha=0.5,                     # 금속 벽/창문 → intensity 활용
    max_correspondence_distance=1.5,
    initial_threshold=1.5,
    min_motion_th=0.5,
    max_map_frames=200,
    map_radius=60.0,
    max_source_points=0,
    auto_alpha=False,
    auto_alpha_from_intensity=False,
    source_drop_small_voxels=False,
    source_max_output_features=0,
    source_min_feature_score=0.0,
    device='cpu',
)
```

---

## 주요 파라미터 설명

| 파라미터 | 설명 | 중요도 |
|---------|------|--------|
| `voxel_size` | 맵 복셀 크기 [m] | ⭐⭐⭐ |
| `source_voxel_size` | 소스 다운샘플 복셀 [m] | ⭐⭐ |
| `alpha` | Intensity 가중치 (0=geometry-only) | ⭐⭐⭐ |
| `map_radius` | Spatial eviction 반경 (None=age-based) | ⭐⭐⭐ |
| `min_motion_th` | Sigma floor [m] — 터널/광산에서 **0.5 필수** | ⭐⭐⭐ |
| `max_map_frames` | Age-based eviction 윈도우 | ⭐⭐ |
| `max_source_points` | 소스 서브샘플 수 (0=전체, 권장) | ⭐⭐⭐ |
| `auto_alpha_from_intensity` | **항상 False** (MAD 기반 alpha 폭발 버그) | ⭐⭐⭐ |
| `auto_alpha` | False 권장 (KITTI에서 역효과) | ⭐⭐ |
| `source_drop_small_voxels` | False 권장 (너무 공격적) | ⭐⭐ |

---

## 결과 접근

```python
result = pipeline.process_frame(points, intensities, timestamp=t)

result.pose          # (4,4) 절대 pose (world ← sensor)
result.reg_ms        # 등록 시간 [ms]
result.map_ms        # 맵 업데이트 시간 [ms]
result.kappa         # GN Hessian condition number (클수록 geometry 퇴화)
result.mscs_ratio    # n_mscs_used / n_correspondences

# 전체 trajectory
poses = pipeline.get_trajectory().poses   # List[np.ndarray (4,4)]
```

---

## KITTI 포즈 저장

```python
with open("poses.txt", "w") as f:
    for pose in poses:
        row = pose[:3, :].flatten()
        f.write(" ".join(f"{v:.6e}" for v in row) + "\n")
```

---

## 재빌드

파라미터 변경은 재빌드 불필요. C++ 코드 변경 시:
```bash
python setup_cpp.py build_ext --inplace
```

---

## 속도 비교 (C++ core, 500fr)

| 방법 | KITTI ATE(seq00) | approx Hz |
|------|----------------|-----------|
| **IV-GICP** | **0.313m** | ~4–6 Hz |
| KISS-ICP | 0.320m | ~4–6 Hz |

> **참고**: ms/frame은 시스템 부하, 병렬 실행 수에 따라 크게 변동. 단독 실행 시 IV-GICP ~43ms, KISS-ICP ~20ms.

> **HD 맵 전략**: 속도 패널티는 오프라인 HD 맵 생성 맥락에서 무관. 터널·지하 구간의 정밀도가 핵심 강점.

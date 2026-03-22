# IV-GICP

**Information-Theoretic Adaptive Voxelization and Intensity-Augmented GICP**

ICRA 논문용 LiDAR odometry 구현. 기하+강도 공분산 4D 정합으로 터널/지하에서 KISS-ICP 대비 최대 49% ATE 개선.

## 설치

```bash
uv sync
python setup_cpp.py build_ext --inplace
```

## 빠른 시작

```python
from iv_gicp.pipeline import IVGICPPipeline

pipeline = IVGICPPipeline(
    voxel_size=1.0,
    source_voxel_size=0.3,
    alpha=0.1,               # 야외; 콘크리트 터널=0.0, metro=0.5
    max_map_frames=500,
    map_radius=None,         # 터널에선 80-200m
    min_motion_th=0.1,       # 터널/광산에선 0.5
    max_source_points=0,     # 0 = 전체 (ATE 최적)
    auto_alpha_from_intensity=False,
    source_drop_small_voxels=False,
    device='cpu',
)

for points, intensities in scan_loader():  # (N,3), (N,) float64 [0,1]
    result = pipeline.process_frame(points, intensities)
    pose = result.pose  # (4,4) 절대 pose
```

## 성능 (500fr, C++ core)

### KITTI (야외 자율주행) — 11개 시퀀스 7/11 IV-GICP 승

| Seq | IV-GICP | KISS-ICP | Δ% |
|-----|---------|---------|-----|
| 00  | **0.313m** | 0.320m | -2.2% |
| 01  | 3.222m | **3.119m** | +3.3% |
| 02  | **0.615m** | 0.807m | -23.8% |
| 05  | **0.351m** | 0.380m | -7.6% |
| 08  | 2.985m | 2.963m | +0.7% |

### GEODE Urban Tunnel (콘크리트 터널) — 3/3 IV-GICP 승 🔥

| Seq | IV-GICP | KISS-ICP | Δ% |
|-----|---------|---------|-----|
| Urban_Tunnel01 | **2.706m** | 4.396m | **-38.4%** |
| Urban_Tunnel02 | **4.152m** | 8.085m | **-48.7%** |
| Urban_Tunnel03 | **12.528m** | 13.808m | -9.3% |

### SubT-MRS (지하/광산) — 3/3 IV-GICP 승

| Dataset | IV-GICP | KISS-ICP | Δ% |
|---------|---------|---------|-----|
| Urban_UGV1 | **0.276m** | 0.285m | -3.2% |
| Urban_UGV2 | **0.280m** | 0.288m | -2.8% |
| Final_UGV1 | **0.084m** | 0.088m | -4.5% |

## 벤치마크 실행

```bash
# KITTI (seq 지정, 500fr)
uv run python examples/run_kitti_benchmark.py --seq 00 --max-frames 500 --skip-genz

# GEODE Urban Tunnel
uv run python examples/run_geode_eval.py --seq 01 --max-frames 500

# SubT-MRS
uv run python examples/run_subt_eval.py --dataset Urban_UGV1 --max-frames 500

# Metro Tunnel
uv run python examples/run_metro_eval.py --seq 1 --max-frames 500

# 전체 (병렬)
uv run python examples/run_full_eval.py --dataset kitti
```

## 문서

| 파일 | 내용 |
|------|------|
| [docs/USAGE.md](docs/USAGE.md) | 파라미터 가이드, 도메인별 최적 설정 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 파이프라인 구조, C++ 모듈 설명 |
| [docs/THEORY.md](docs/THEORY.md) | FIM 프레임워크, Theorem 1, MSCS 이론 |

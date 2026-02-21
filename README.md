# IV-GICP

**Information-Theoretic Adaptive Voxelization and Intensity-Augmented GICP with Retroactive Map Distribution Refinement**

ICRA 논문용 LiDAR Odometry 구현.

## 요약

- **Adaptive Voxelization**: Shannon entropy + intensity variance 기반 octree 분할
- **IV-GICP**: 4D geo-photometric 확률 정합 (기하+광도 공분산)
- **Factor Graph**: Fixed-lag pose graph 최적화 (FORM 스타일)
- **FORM Map**: 점 단위 갱신 `p_new = T_new @ T_old^{-1} @ p` (baseline)
- **Distribution Propagation**: 복셀 (μ, Σ)만 Lie theory로 업데이트 (FORM 대비 O(k) vs O(n))

## 설치

### uv (권장)

```bash
# 의존성 설치
uv sync

# 실행
uv run python examples/run_iv_gicp.py --frames 20
uv run python examples/run_form_benchmark.py

# Figure + Synthetic 데이터 + GT 비교 (기본: output/)
uv run python examples/run_iv_gicp.py --frames 20
# → output/data/pointclouds/, poses_gt.npy, metrics.json
# → output/trajectory_2d.png (GT vs IV-GICP 비교)

# 저장된 데이터로 재실험 (재현성)
uv run python examples/run_iv_gicp.py --load-data output/data

# map_3d: DISPLAY 있으면 Open3D, 없으면 matplotlib
uv run python examples/run_iv_gicp.py --frames 20 --no-open3d-3d

# 저장 안 함
uv run python examples/run_iv_gicp.py --frames 20 --no-save

# KITTI Raw Data
uv run python examples/run_kitti.py --data data/kitti/sample -o output/kitti
uv run python examples/run_kitti.py --data data/kitti/sample --max-frames 50 --downsample 5
# → output/kitti/report.json, report.txt, metrics.json, poses_est.npy
```

### pip

```bash
pip install -r requirements.txt
```

## 사용법

```python
from iv_gicp import IVGICPPipeline
import numpy as np

# (N, 4) 포인트 클라우드 [x, y, z, intensity]
points = np.random.randn(5000, 4)
points[:, :3] *= 5
points[:, 3] = np.random.rand(5000) * 100

pipeline = IVGICPPipeline(
    voxel_size=0.5,
    entropy_threshold=2.0,
    intensity_var_threshold=50.0,
)
result = pipeline.process_frame(points)
print(result.pose)
```

## 예제 실행

```bash
# 합성 데이터
python examples/run_iv_gicp.py --frames 20

# PCD 파일
python examples/run_iv_gicp.py --pcd path/to/scan.pcd
```

## 논문 구조

`docs/` 폴더의 MD 파일 참고.

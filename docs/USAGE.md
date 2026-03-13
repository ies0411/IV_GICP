# IV-GICP — Usage Guide

**IV-GICP** is a LiDAR odometry library that processes point clouds sequentially and returns SE(3) poses. It works like KISS-ICP: feed scans one at a time, get poses out.

---

## Installation

```bash
# Clone and install
git clone https://github.com/your-repo/iv_gicp
cd iv_gicp
uv sync   # or: pip install -e .

# Build C++ extension (recommended — ~25 Hz vs ~5 Hz without it)
uv run python setup_cpp.py build_ext --inplace


uv run python examples/run_kitti.py \
  --format odometry \
  --data /home/km/data/kitti/dataset \
  --seq 00 \
  --max-frames 100
```

---

## Quick Start

```python
import numpy as np
from iv_gicp import IVGICPOdometry

od = IVGICPOdometry()

for points in your_scan_sequence:   # points: (N, 4) [x, y, z, intensity]
    pose = od.register_frame(points)
    print(pose)   # (4, 4) SE(3) transform — world ← sensor

# All poses at once
poses = od.poses          # list of (4, 4) numpy arrays
last  = od.last_pose      # most recent pose
times = od.timestamps     # list of float timestamps
```

`points` can be:
- **(N, 4)** — `[x, y, z, intensity]` (preferred)
- **(N, 3)** — `[x, y, z]` only (geometry-only mode, sets `alpha=0` recommended)

---

## Preset Configs

Use a preset that matches your sensor environment. All presets use the best-validated parameters from our evaluation (see [eval.md](eval.md)).

```python
from iv_gicp import IVGICPOdometry

# Outdoor driving (KITTI, urban)  — default
od = IVGICPOdometry.outdoor()

# Indoor rooms / hallways (Hilti, SubT mines)
od = IVGICPOdometry.indoor()

# Tunnels, mines, long corridors with degenerate geometry (GEODE Urban Tunnel)
od = IVGICPOdometry.tunnel()

# Subway / metro with loop-like structure (GEODE Metro)
od = IVGICPOdometry.metro()

# Underground mines (SubT, MulRan)
od = IVGICPOdometry.underground()

# GPU acceleration (all presets accept device=)
od = IVGICPOdometry.outdoor(device="cuda")
```

### When to use which preset

| Environment | Preset | Key behaviour |
|------------|--------|---------------|
| Large-scale outdoor, varied geometry | `outdoor()` | `voxel=1.0 α=0.1` age-based map |
| Indoor structured rooms | `indoor()` | `voxel=0.3 α=0.5` spatial eviction 40 m |
| Tunnels / mines (degenerate geometry) | `tunnel()` | `voxel=0.5 α=0.0` spatial eviction 80 m |
| Subway / metro | `metro()` | `voxel=0.5 α=0.5` window smoothing W=10 |
| Underground with open areas | `underground()` | `voxel=0.5 α=0.1` spatial eviction 50 m |

---

## Custom Parameters

```python
from iv_gicp import IVGICPOdometry

od = IVGICPOdometry(
    voxel_size=0.5,               # map voxel size [m]
    alpha=0.3,                    # intensity weight (0 = pure geometry)
    source_voxel_size=0.2,        # input scan downsampling [m]; 0 = disabled
    max_correspondence_distance=1.5,  # max ICP match distance [m]
    min_range=0.5,                # ignore returns closer than this [m]
    max_range=60.0,               # ignore returns farther than this [m]
    map_radius=60.0,              # spatial eviction radius [m]; None = age-based
    max_map_frames=30,            # how many recent frames to keep; None = auto
    window_size=1,                # FORM window smoothing (1 = disabled)
    adaptive_voxelization=False,  # C1 entropy split (slower; helps tunnels)
    device="auto",                # "auto" | "cuda" | "cpu"
)
```

### Key parameters explained

| Parameter | Effect |
|-----------|--------|
| `voxel_size` | Map resolution. Larger = faster but less accurate on fine structure. |
| `alpha` | Intensity weight. `0` = geometry only; `0.5` = strong intensity guidance. Use `>0` when geometry is degenerate (tunnels, corridors). |
| `source_voxel_size` | Downsamples each input scan before registration. Reduces noise and speeds up matching. |
| `map_radius` | Spatial eviction: drops voxels farther than `R` metres from current position. Better than age-based in long corridors. `None` uses age-based eviction (better for outdoors with U-turns). |
| `window_size` | FORM-style joint optimisation over the last `K` frames. Helps degenerate sequences. `1` = disabled. |
| `adaptive_voxelization` | Entropy-based C1 splitting. Automatically uses small voxels in complex geometry, large voxels in flat areas. Slower than the C++ flat map. |

---

## Reading KITTI

```python
import numpy as np
from pathlib import Path
from iv_gicp import IVGICPOdometry
from iv_gicp.kitti_loader import load_kitti_odometry_sequence

frames, gt_poses, timestamps = load_kitti_odometry_sequence(
    data_root="/data/kitti/dataset",
    seq="00",
    max_frames=500,
)

od = IVGICPOdometry.outdoor(device="cuda")

for pts, ts in zip(frames, timestamps):
    pose = od.register_frame(pts, timestamp=ts)
```

---

## Providing Timestamps

Timestamps are optional but stored in `od.timestamps` if provided.

```python
pose = od.register_frame(points, timestamp=1712345678.123)
```

---

## Separating Points and Intensities

If your pipeline keeps geometry and intensity separate:

```python
xyz  = scan[:, :3]          # (N, 3)
ints = scan[:, 3]           # (N,)

pose = od.register_frame(xyz, intensities=ints)
```

---

## Saving Poses (KITTI format)

```python
with open("poses.txt", "w") as f:
    for T in od.poses:
        row = T[:3, :].flatten()          # 12 values (3×4)
        f.write(" ".join(f"{v:.6e}" for v in row) + "\n")
```

---

## Advanced: Access the Full Pipeline

For low-level control (map inspection, ablation configs, etc.):

```python
pipeline = od.pipeline   # IVGICPPipeline instance

# Example: check map size
if pipeline._cpp_voxel_map is not None:
    print("C++ map voxels:", pipeline._cpp_voxel_map.num_voxels())
```

---

## Performance Expectations

| Mode | Hardware | Speed | Notes |
|------|----------|-------|-------|
| C++ core (default) | CPU | ~25 Hz | Requires `setup_cpp.py build_ext --inplace` |
| Python / CUDA | GPU | ~1 Hz | Fallback without C++ build |
| KISS-ICP (reference) | CPU | ~37 Hz | C++ baseline |

Build the C++ extension for real-time use.

---

## Accuracy (KITTI 500 frames)

| Sequence | IV-GICP | KISS-ICP | Δ |
|----------|---------|---------|---|
| seq00 | 0.299 m | 0.320 m | **−6.6%** |
| seq05 | 0.366 m | 0.380 m | **−3.7%** |
| seq08 | 3.040 m | 2.963 m | +2.6% (hilly terrain) |

See [eval.md](eval.md) for full results across KITTI, SubT, GEODE, HeLiPR, MulRan.

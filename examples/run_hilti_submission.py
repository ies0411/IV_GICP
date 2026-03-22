"""
Hilti SLAM Challenge — Submission Generator
============================================
Runs IV-GICP on all available Hilti 2021/2022 sequences and produces
submission-format trajectory files.

Hilti submission format (TUM-like):
    timestamp tx ty tz qx qy qz qw  (space-separated, one pose per LiDAR frame)

Output:
    results/hilti/2021/<seq_name>.txt   — submission files
    results/hilti/2022/<seq_name>.txt
    results/hilti/submission_2021.zip
    results/hilti/submission_2022.zip

Best params (CLAUDE.md, Hilti corridor):
    voxel=0.3, source=0.2, alpha=0.5, mc=0.5, map_radius=40m, C++ VoxelMap

Usage:
    uv run python examples/run_hilti_submission.py
    uv run python examples/run_hilti_submission.py --year 2022
    uv run python examples/run_hilti_submission.py --seq exp07_long_corridor --year 2022
"""

import argparse
import json
import time
import zipfile
from pathlib import Path
import numpy as np

# ── Dataset config ─────────────────────────────────────────────────────────────

DATASETS = {
    "2021": {
        "base": "/home/km/data/hilti/2021",
        "lidar_topic": "/os_cloud_node/points",
        "parser": "ouster",   # Ouster OS0/OS1
        "sequences": [
            "Basement_1",
            "Basement_3",
            "exp07_long_corridor",
            "exp14_basement_2",
            "exp18_corridor_lower_gallery_2",
        ],
    },
    "2022": {
        "base": "/home/km/data/hilti/2022",
        "lidar_topic": "/hesai/pandar",
        "parser": "hesai",    # Hesai Pandar64
        "sequences": [
            "exp07_long_corridor",
            "exp14_basement_2",
            "exp18_corridor_lower_gallery_2",
        ],
    },
}

# Best params from CLAUDE.md (Hilti corridor: tight indoor, degenerate corridors)
IV_GICP_PARAMS = dict(
    voxel_size=0.3,
    source_voxel_size=0.2,
    alpha=0.5,                         # intensity important in degenerate corridor
    max_correspondence_distance=0.5,   # tight: slow indoor robot
    initial_threshold=0.15,
    min_motion_th=0.03,
    max_iterations=20,
    map_radius=40.0,                   # spatial eviction: 40m bubble
)


# ── LiDAR parsers ──────────────────────────────────────────────────────────────

def _parse_hesai(msg) -> np.ndarray:
    """Hesai Pandar64 PointCloud2 → (N,4) float64 [x,y,z,intensity]"""
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    x   = data[:, 0:4].view(np.float32).reshape(-1)
    y   = data[:, 4:8].view(np.float32).reshape(-1)
    z   = data[:, 8:12].view(np.float32).reshape(-1)
    itn = data[:, 16:20].view(np.float32).reshape(-1)
    pts = np.stack([x, y, z, itn], axis=1).astype(np.float64)
    r   = np.linalg.norm(pts[:, :3], axis=1)
    return pts[np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < 80.0)]


def _parse_ouster(msg) -> np.ndarray:
    """Ouster OS0/OS1 PointCloud2 → (N,4) float64 [x,y,z,intensity]
    Field offsets: x=0, y=4, z=8, intensity=16, point_step=48 (float32 each)
    """
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    x   = data[:, 0:4].view(np.float32).reshape(-1)
    y   = data[:, 4:8].view(np.float32).reshape(-1)
    z   = data[:, 8:12].view(np.float32).reshape(-1)
    itn = data[:, 16:20].view(np.float32).reshape(-1)
    pts = np.stack([x, y, z, itn], axis=1).astype(np.float64)
    r   = np.linalg.norm(pts[:, :3], axis=1)
    return pts[np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < 80.0)]


PARSERS = {"hesai": _parse_hesai, "ouster": _parse_ouster}


# ── Bag reader ─────────────────────────────────────────────────────────────────

def read_bag(bag_path: str, topic: str, parser_name: str, max_frames: int = None):
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.ROS1_NOETIC)
    parse_fn  = PARSERS[parser_name]

    frames = []
    print(f"  Reading: {Path(bag_path).name}")
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == topic]
        total = sum(1 for _ in bag.messages(connections=conns))
        conns = [c for c in bag.connections if c.topic == topic]
        for conn, ts_ns, raw in bag.messages(connections=conns):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            pts = parse_fn(msg)
            if pts is not None and len(pts) > 200:
                frames.append((ts_ns / 1e9, pts))
                if len(frames) % 200 == 0:
                    print(f"  {len(frames)}/{total} frames loaded ...", end="\r")
                if max_frames and len(frames) >= max_frames:
                    break
    dur = frames[-1][0] - frames[0][0] if len(frames) > 1 else 0.0
    avg_pts = float(np.mean([len(f[1]) for f in frames]))
    print(f"  {len(frames)} frames ({dur:.1f}s, avg {avg_pts:.0f} pts/frame)    ")
    return frames


# ── IV-GICP runner ─────────────────────────────────────────────────────────────

def run_iv_gicp(frames, device="cuda"):
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from iv_gicp.pipeline import IVGICPPipeline

    pipeline = IVGICPPipeline(device=device, **IV_GICP_PARAMS)

    abs_poses = [np.eye(4)]
    times = []
    n = len(frames)
    print(f"  [IV-GICP] {n} frames, device={device}, map_radius={IV_GICP_PARAMS['map_radius']}m")

    for i, (ts, pts) in enumerate(frames):
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        elapsed = (time.perf_counter() - t0) * 1000
        abs_poses.append(result.pose.copy())
        times.append(elapsed)
        if i % 100 == 0 or i == n - 1:
            print(f"  {i+1:4d}/{n}  {elapsed:6.1f}ms", end="\r")

    times = np.array(times[1:])  # skip first frame
    print(f"\n  mean={times.mean():.1f}ms  median={np.median(times):.1f}ms  "
          f"→ {1000/times.mean():.2f} Hz")
    return abs_poses


# ── Pose utilities ────────────────────────────────────────────────────────────

def save_submission_tum(poses, timestamps, path: Path):
    """Save in Hilti submission format: timestamp tx ty tz qx qy qz qw"""
    from scipy.spatial.transform import Rotation
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
            f.write(f"{ts:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
    print(f"  Saved: {path}")


def traj_stats(poses):
    dists = [np.linalg.norm(poses[i+1][:3, 3] - poses[i][:3, 3])
             for i in range(len(poses) - 1)]
    return float(np.sum(dists)), float(np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",  choices=["2021", "2022", "all"], default="all")
    parser.add_argument("--seq",   default=None, help="Run single sequence by name")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    years = ["2021", "2022"] if args.year == "all" else [args.year]
    out_root = Path(__file__).parent.parent / "results" / "hilti"
    all_results = {}

    for year in years:
        cfg = DATASETS[year]
        seqs = [args.seq] if args.seq else cfg["sequences"]
        out_dir = out_root / year
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Hilti {year}  |  {len(seqs)} sequences  |  device={device}")
        print(f"{'='*60}")

        year_results = {}
        for seq in seqs:
            bag_path = f"{cfg['base']}/{seq}.bag"
            if not Path(bag_path).exists():
                print(f"  [SKIP] {bag_path} not found")
                continue

            sub_path_check = out_dir / f"{seq}.txt"
            if sub_path_check.exists() and sub_path_check.stat().st_size > 0:
                print(f"  [SKIP] {seq} already done ({sub_path_check.stat().st_size//1024}KB)")
                year_results[seq] = {"n_frames": sum(1 for _ in open(sub_path_check)),
                                     "path_length_m": 0, "end_displacement_m": 0,
                                     "submission_file": str(sub_path_check)}
                continue

            print(f"\n[{seq}]")
            frames = read_bag(bag_path, cfg["lidar_topic"], cfg["parser"], args.max_frames)
            timestamps = [f[0] for f in frames]

            poses = run_iv_gicp(frames, device=device)

            # Save submission file
            sub_path = out_dir / f"{seq}.txt"
            save_submission_tum(poses, timestamps, sub_path)

            path_len, end_disp = traj_stats(poses)
            year_results[seq] = {
                "n_frames": len(frames),
                "path_length_m": path_len,
                "end_displacement_m": end_disp,
                "submission_file": str(sub_path),
            }

        # Create zip for submission
        if year_results:
            zip_path = out_root / f"submission_{year}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for seq, r in year_results.items():
                    txt = Path(r["submission_file"])
                    zf.write(txt, txt.name)
            print(f"\n  Submission zip: {zip_path}")

        all_results[year] = year_results

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for year, yr in all_results.items():
        for seq, r in yr.items():
            print(f"  {year}/{seq:<42}  {r['n_frames']:>5}fr  "
                  f"path={r['path_length_m']:.1f}m  endDisp={r['end_displacement_m']:.2f}m")

    # Save params + results
    meta = {"iv_gicp_params": IV_GICP_PARAMS, "device": device, "results": all_results}
    with open(out_root / "submission_results.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Params + stats: {out_root}/submission_results.json")
    print(f"  Submit zips from: {out_root}/")


if __name__ == "__main__":
    main()

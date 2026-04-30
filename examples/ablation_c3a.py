#!/usr/bin/env python3
"""
C3-A Ablation: per-voxel adaptive alpha validation.

Tests the new C3-A implementation on GEODE Urban Tunnel 01 (500fr).
Key question: does per-voxel adaptive alpha fix C1+C2 destructive interference?

Prior ablation (fixed alpha):
  A: Base   = 1.064m
  B: +C1    = 0.821m  (-22.8%) ← C1 alone helps
  C: +C2    = 1.081m  (+1.6%)  ← C2 alone neutral
  D: C1+C2  = 1.506m  (+41.6%) ← C1+C2 interference!

Expected with C3-A:
  D':  C2+C3-A        = better than +C2  (alpha near 0 for well-conditioned voxels)
  E':  C1+C2+C3-A     = ~B (+C1) level or better (C1 and C2 now complementary)

Usage:
    uv run python examples/ablation_c3a.py --seq 01 --max-frames 500
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

GEODE_ROOT = Path("/home/km/deepai_dev_data/GEODE")
LIDAR_TOPIC = "/velodyne_points"


def read_geode_frames(bag_path: Path, max_frames: int = None):
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.ROS1_NOETIC)
    frames = []
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        for conn, ts_ns, rawdata in bag.messages(connections=conns):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
            x   = data[:, 0:4].view(np.float32).reshape(-1)
            y   = data[:, 4:8].view(np.float32).reshape(-1)
            z   = data[:, 8:12].view(np.float32).reshape(-1)
            itn = data[:, 12:16].view(np.float32).reshape(-1)
            pts = np.stack([x, y, z, itn], axis=1).astype(np.float64)
            r   = np.linalg.norm(pts[:, :3], axis=1)
            pts = pts[np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < 80.0)]
            if len(pts) > 100:
                frames.append((ts_ns / 1e9, pts))
            if max_frames and len(frames) >= max_frames:
                break
    return frames


def load_geode_gt(gt_path: Path):
    from scipy.spatial.transform import Rotation
    data = np.loadtxt(gt_path)
    T0_inv = None
    ts, poses = [], []
    for row in data:
        R = Rotation.from_quat(row[4:8]).as_matrix()
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = row[1:4]
        if T0_inv is None: T0_inv = np.linalg.inv(T)
        poses.append(T0_inv @ T)
        ts.append(row[0])
    return np.array(ts), np.stack(poses)


def compute_ate_umeyama(est_poses, est_ts, gt_ts, gt_poses):
    matched_e, matched_g = [], []
    for i, ts in enumerate(est_ts):
        idx = np.argmin(np.abs(gt_ts - ts))
        if abs(gt_ts[idx] - ts) < 0.2:
            matched_e.append(est_poses[i][:3, 3])
            matched_g.append(gt_poses[idx][:3, 3])
    if len(matched_e) < 5: return float('nan')
    P = np.array(matched_e).T
    Q = np.array(matched_g).T
    mu_p = P.mean(1, keepdims=True); mu_q = Q.mean(1, keepdims=True)
    H = (P - mu_p) @ (Q - mu_q).T
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_q - R @ mu_p
    return float(np.sqrt(np.mean(np.linalg.norm(R @ P + t - Q, axis=0)**2)))


def run_config(frames, label, **kwargs):
    from iv_gicp.pipeline import IVGICPPipeline
    # Fixed base params for GEODE Urban Tunnel
    base = dict(
        voxel_size=0.5,
        source_voxel_size=0.25,
        max_correspondence_distance=2.0,
        initial_threshold=1.5,
        min_motion_th=0.5,
        max_iterations=12,
        max_map_frames=500,
        map_radius=80.0,
        auto_alpha=False,
        auto_alpha_from_intensity=False,
        source_drop_small_voxels=False,
        source_max_output_features=0,
        source_min_feature_score=0.0,
        max_source_points=0,
        device="cpu",
    )
    base.update(kwargs)
    pipeline = IVGICPPipeline(**base)
    poses, times = [], []
    for i, (ts, pts) in enumerate(frames):
        t0 = time.perf_counter()
        r = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        times.append((time.perf_counter() - t0) * 1000)
        poses.append(r.pose.copy())
        if (i + 1) % 100 == 0:
            print(f"  [{label}] {i+1}/{len(frames)}  {np.mean(times[-50:]):.1f}ms/fr", end="\r")
    print(f"  [{label}] {len(frames)}/{len(frames)}  mean={np.mean(times[1:]):.1f}ms/fr     ")
    return poses, times


def find_geode_bag(seq_id: str):
    seq_name = f"Urban_Tunnel{seq_id.zfill(2)}"
    candidates = [
        (GEODE_ROOT / seq_name / f"{seq_name}.bag",
         GEODE_ROOT / seq_name / "gt.txt"),
        (GEODE_ROOT / "sensor_data" / "Urban_tunnel" / seq_name / f"{seq_name}.bag",
         GEODE_ROOT / "groundtruth" / "Urban_tunnel" / f"{seq_name}.txt"),
    ]
    for bag, gt in candidates:
        if bag.exists():
            return bag, gt
    raise FileNotFoundError(f"GEODE seq {seq_id} not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="01", choices=["01", "02", "03"])
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--kappa0", type=float, default=50.0,
                    help="C3-A kappa0 threshold (entropy_scale_c)")
    args = ap.parse_args()

    seq_name = f"Urban_Tunnel{args.seq.zfill(2)}"
    bag_path, gt_path = find_geode_bag(args.seq)

    print(f"[C3-A Ablation] {seq_name}  {args.max_frames}fr  kappa0={args.kappa0}")
    print(f"  Loading frames from {bag_path.name}...")
    frames = read_geode_frames(bag_path, args.max_frames)
    print(f"  {len(frames)} frames loaded")
    gt_ts, gt_poses = load_geode_gt(gt_path)
    scan_ts = np.array([f[0] for f in frames])

    configs = [
        # label,  alpha,  use_fim_weight, use_entropy_alpha, entropy_scale_c
        ("A: Base GICP",      0.0,  False, False, 50.0),
        ("B: +C1",            0.0,  True,  False, 50.0),
        ("C: +C2 (alpha=0.1)",0.1,  False, False, 50.0),
        ("D: C1+C2",          0.1,  True,  False, 50.0),
        ("D': C2+C3-A",       0.1,  False, True,  args.kappa0),
        ("E': C1+C2+C3-A",    0.1,  True,  True,  args.kappa0),
        ("F: C1(alpha=0)",    0.0,  True,  True,  args.kappa0),  # C1 + C3-A (alpha=0 → C3-A has no effect)
    ]

    results = {}
    for label, alpha, fim_w, entropy_a, k0 in configs:
        print(f"\n  Running: {label}")
        try:
            poses, times = run_config(
                frames, label,
                alpha=alpha,
                use_fim_weight=fim_w,
                use_entropy_alpha=entropy_a,
                entropy_scale_c=k0,
            )
            ate = compute_ate_umeyama(poses, scan_ts, gt_ts, gt_poses)
            results[label] = ate
            print(f"  → ATE = {ate:.4f}m")
        except Exception as e:
            print(f"  → FAILED: {e}")
            import traceback; traceback.print_exc()
            results[label] = float('nan')

    # Summary
    base_ate = results.get("A: Base GICP", float('nan'))
    print(f"\n{'='*65}")
    print(f"C3-A Ablation: {seq_name}  {args.max_frames}fr  kappa0={args.kappa0}")
    print(f"{'Config':<28} {'ATE(m)':>8}  {'vs Base':>8}")
    print(f"{'-'*55}")
    for label, ate in results.items():
        if np.isfinite(ate) and np.isfinite(base_ate):
            delta = (ate - base_ate) / base_ate * 100
            marker = "✓" if delta < 0 else ("✗" if delta > 5 else "~")
            print(f"  {label:<26} {ate:>8.4f}m  {delta:>+7.1f}%  {marker}")
        else:
            print(f"  {label:<26} {'nan':>8}   {'nan':>8}")

    # Key comparison
    if "D: C1+C2" in results and "E': C1+C2+C3-A" in results:
        d = results["D: C1+C2"]
        e = results["E': C1+C2+C3-A"]
        if np.isfinite(d) and np.isfinite(e):
            improv = (d - e) / d * 100
            print(f"\n  C3-A improvement over C1+C2: {improv:+.1f}%")
            print(f"  (D={d:.4f}m → E'={e:.4f}m)")

    import json
    out = Path(__file__).parent.parent / "results" / "geode" / seq_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "c3a_ablation.json").write_text(
        json.dumps({"seq": seq_name, "frames": args.max_frames,
                    "kappa0": args.kappa0, "ate": results}, indent=2)
    )
    print(f"\nSaved → {out}/c3a_ablation.json")


if __name__ == "__main__":
    main()

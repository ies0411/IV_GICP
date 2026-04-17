"""
Hilti SLAM Challenge — IV-GICP vs KISS-ICP vs GenZ-ICP
=======================================================
Reads Hilti bags once, runs all three methods, evaluates against local GT.

GT is sparse (5–6 prism/pole measurements per sequence).
Evaluation: Umeyama alignment + APE, same as official Hilti eval kit.

Usage:
    uv run python examples/bench_hilti_comparison.py
    uv run python examples/bench_hilti_comparison.py --seq exp07
    uv run python examples/bench_hilti_comparison.py --local  # use /tmp cached bags
"""
import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

BASE    = Path(__file__).parent.parent
RUNNER  = BASE / "thirdparty/genz-icp/bin_runner/build/genz_bin_runner"
OUT_DIR = BASE / "results/hilti/comparison"

# ── Sequence configs ───────────────────────────────────────────────────────────

SEQUENCES = {
    "exp07": {
        "year":    "2022",
        "bag":     "/home/km/deepai_dev_data/hilti/2022/exp07_long_corridor.bag",
        "bag_local": "/tmp/hilti_exp07.bag",
        "topic":   "/hesai/pandar",
        "parser":  "hesai",
        "gt":      str(BASE / "thirdparty/hilti-slam-challenge-2022/groundtruth_2022/exp07_long_corridor.txt"),
        "iv_traj": str(BASE / "results/hilti/2022/exp07_long_corridor.txt"),
        # Hesai Pandar64 — alpha=0.5 gives good corridor intensity
        "iv_params": dict(voxel_size=0.3, source_voxel_size=0.2, alpha=0.5,
                          max_correspondence_distance=0.5, initial_threshold=0.5,
                          min_motion_th=0.5, max_iterations=20, map_radius=40.0,
                          min_range=0.5, max_range=40.0),
        "kiss_voxel": 0.3,
        "genz_params": dict(voxel_size=0.3, max_range=40.0, min_range=0.5,
                            min_motion_th=0.5, initial_threshold=0.5, planarity_th=0.1),
    },
    "Basement_1": {
        "year":    "2021",
        "bag":     "/home/km/deepai_dev_data/hilti/2021/Basement_1.bag",
        "bag_local": "/tmp/hilti_b1.bag",
        "topic":   "/os_cloud_node/points",
        "parser":  "ouster",
        "gt":      str(BASE / "thirdparty/hilti-slam-challenge-2021/evaluation-evo/data/Basement_1_pole.txt"),
        "iv_traj": str(BASE / "results/hilti/2021/Basement_1.txt"),
        # Ouster OS1-64 — alpha=0.1
        "iv_params": dict(voxel_size=0.3, source_voxel_size=0.2, alpha=0.1,
                          max_correspondence_distance=0.5, initial_threshold=0.5,
                          min_motion_th=0.5, max_iterations=20, map_radius=40.0,
                          min_range=0.5, max_range=40.0),
        "kiss_voxel": 0.3,
        "genz_params": dict(voxel_size=0.3, max_range=40.0, min_range=0.5,
                            min_motion_th=0.5, initial_threshold=0.5, planarity_th=0.1),
    },
}

# ── LiDAR parsers ──────────────────────────────────────────────────────────────

def _parse_hesai(msg) -> np.ndarray:
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    x   = data[:, 0:4].view(np.float32).reshape(-1)
    y   = data[:, 4:8].view(np.float32).reshape(-1)
    z   = data[:, 8:12].view(np.float32).reshape(-1)
    itn = data[:, 16:20].view(np.float32).reshape(-1)
    pts = np.stack([x, y, z, itn], axis=1).astype(np.float64)
    r   = np.linalg.norm(pts[:, :3], axis=1)
    return pts[np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < 80.0)]

def _parse_ouster(msg) -> np.ndarray:
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    x   = data[:, 0:4].view(np.float32).reshape(-1)
    y   = data[:, 4:8].view(np.float32).reshape(-1)
    z   = data[:, 8:12].view(np.float32).reshape(-1)
    itn = data[:, 16:20].view(np.float32).reshape(-1)
    pts = np.stack([x, y, z, itn], axis=1).astype(np.float64)
    r   = np.linalg.norm(pts[:, :3], axis=1)
    return pts[np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < 80.0)]

PARSERS = {"hesai": _parse_hesai, "ouster": _parse_ouster}


# ── Bag reader (single-pass) ──────────────────────────────────────────────────

def read_bag(bag_path: str, topic: str, parser_name: str):
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.ROS1_NOETIC)
    parse_fn  = PARSERS[parser_name]
    frames = []
    t0 = time.perf_counter()
    print(f"  Reading: {Path(bag_path).name}", flush=True)
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == topic]
        for conn, ts_ns, raw in bag.messages(connections=conns):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            pts = parse_fn(msg)
            if pts is not None and len(pts) > 200:
                frames.append((ts_ns / 1e9, pts))
                if len(frames) % 200 == 0:
                    print(f"  {len(frames)} frames ...", end="\r", flush=True)
    dur  = frames[-1][0] - frames[0][0] if len(frames) > 1 else 0
    avg  = float(np.mean([len(f[1]) for f in frames]))
    read_t = time.perf_counter() - t0
    print(f"  {len(frames)} frames ({dur:.1f}s data, avg {avg:.0f} pts, load={read_t:.1f}s)")
    return frames


# ── Runners ───────────────────────────────────────────────────────────────────

def run_kiss(frames, voxel_size=0.3, max_range=40.0):
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig()
    cfg.mapping.voxel_size = voxel_size
    cfg.data.max_range = max_range
    cfg.data.min_range = 0.5
    cfg.data.deskew = False  # uniform per-frame timestamps → deskew causes NaN
    od = KissICP(config=cfg)

    poses = []
    timestamps = []
    times = []
    n = len(frames)
    print(f"  [KISS-ICP] voxel={voxel_size}", flush=True)
    for i, (ts, pts) in enumerate(frames):
        xyz = pts[:, :3].copy()
        # extra safety: finite, no huge coords
        mask = (np.isfinite(xyz).all(axis=1) &
                (np.linalg.norm(xyz, axis=1) < max_range) &
                (np.linalg.norm(xyz, axis=1) > 0.3))
        xyz = xyz[mask]
        if len(xyz) < 100:
            continue
        t0 = time.perf_counter()
        od.register_frame(xyz, np.zeros(len(xyz)))
        times.append((time.perf_counter() - t0) * 1000)
        poses.append(od.last_pose.copy())
        timestamps.append(ts)
        if i % 200 == 0 or i == n - 1:
            print(f"  fr={i+1:4d}/{n}  pts={len(xyz):5d}  {times[-1]:.1f}ms",
                  end="\r", flush=True)
    arr = np.array(times[1:]) if len(times) > 1 else np.array([0.0])
    print(f"\n  {len(poses)} poses  mean={arr.mean():.1f}ms  → {1000/arr.mean():.1f} Hz")
    return poses, timestamps


def run_genz(frames, params: dict, tmp_dir: Path):
    """Save frames as .bin, run genz_bin_runner, return poses list."""
    import struct
    bin_dir = tmp_dir / "bins"
    bin_dir.mkdir(parents=True, exist_ok=True)

    timestamps = []
    for i, (ts, pts) in enumerate(frames):
        # GenZ binary expects 4 floats per point (KITTI format: x,y,z,intensity)
        xyzi = pts[:, :4].astype(np.float32) if pts.shape[1] >= 4 else \
               np.hstack([pts[:, :3].astype(np.float32),
                          np.zeros((len(pts), 1), np.float32)])
        xyzi.tofile(bin_dir / f"{i:06d}.bin")
        timestamps.append(ts)

    poses_file = tmp_dir / "genz_poses.txt"
    args = [str(RUNNER), str(bin_dir), str(poses_file)]
    for k, v in params.items():
        args += [f"--{k}", str(v)]

    print(f"  [GenZ-ICP] voxel={params['voxel_size']}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  GenZ-ICP FAILED: {result.stderr[:200]}")
        return None, timestamps
    n = len(frames)
    print(f"  Done in {elapsed:.1f}s → {n/elapsed:.1f} Hz")

    # Load KITTI-format poses (3x4 row-major per line)
    raw = np.loadtxt(poses_file)
    poses = [np.eye(4)]
    for row in raw:
        T = np.eye(4)
        T[:3, :] = row.reshape(3, 4)
        poses.append(T)
    T0_inv = np.linalg.inv(poses[0])
    poses = [T0_inv @ p for p in poses]
    return poses, timestamps


def run_ivgicp(frames, params: dict):
    import sys; sys.path.insert(0, str(BASE))
    from iv_gicp.pipeline import IVGICPPipeline
    pipeline = IVGICPPipeline(**params)
    poses = [np.eye(4)]
    times = []
    n = len(frames)
    print(f"  [IV-GICP] alpha={params['alpha']}", flush=True)
    for i, (ts, pts) in enumerate(frames):
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        times.append((time.perf_counter() - t0) * 1000)
        poses.append(result.pose.copy())
        if i % 200 == 0 or i == n - 1:
            print(f"  {i+1:4d}/{n}  {times[-1]:.1f}ms", end="\r", flush=True)
    arr = np.array(times[1:])
    print(f"\n  mean={arr.mean():.1f}ms  → {1000/arr.mean():.1f} Hz")
    return poses


# ── TUM I/O ───────────────────────────────────────────────────────────────────

def save_tum(poses, timestamps, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            q = R.from_matrix(T[:3, :3]).as_quat()  # xyzw
            f.write(f"{ts:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


# ── APE evaluation (Hilti-style) ──────────────────────────────────────────────

# Pole-tip calibration: LiDAR frame → prism reference
T_POLE = np.array([[1, 0, 0,  0.059],
                   [0, 1, 0, -0.00855],
                   [0, 0, 1,  0.1964],
                   [0, 0, 0,  1.0]])


def evaluate_ape(tum_file: str, gt_file: str, label: str) -> dict:
    from evo.core import metrics, sync
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    import copy

    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    # Apply pole-tip calibration to estimate
    data = np.loadtxt(tum_file)
    stamps = data[:, 0]
    xyz_out, qwxyz_out = [], []
    for row in data:
        rot = R.from_quat(row[4:8]).as_matrix()
        t   = row[1:4].reshape(3, 1)
        T   = np.vstack([np.hstack([rot, t]), [0, 0, 0, 1]])
        Tc  = T @ T_POLE
        xyz_out.append(Tc[:3, 3])
        q = R.from_matrix(Tc[:3, :3]).as_quat()
        qwxyz_out.append([q[3], q[0], q[1], q[2]])
    traj_est = PoseTrajectory3D(np.array(xyz_out), np.array(qwxyz_out), stamps)

    traj_ref_s, traj_est_s = sync.associate_trajectories(traj_ref, traj_est, max_diff=2.0)
    traj_al = copy.deepcopy(traj_est_s)
    traj_al.align(traj_ref_s, correct_scale=False)

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_ref_s, traj_al))
    st = ape.get_all_statistics()

    path_len = float(np.sum(np.linalg.norm(
        np.diff(np.array(xyz_out), axis=0), axis=1)))

    print(f"  {label:<12}  RMSE={st['rmse']*100:6.2f}cm  "
          f"mean={st['mean']*100:6.2f}cm  max={st['max']*100:6.2f}cm  "
          f"path={path_len:.1f}m")
    return st


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default=None, help="exp07 or Basement_1 (default: both)")
    parser.add_argument("--local", action="store_true", help="Use /tmp cached bags")
    parser.add_argument("--skip-iv", action="store_true", help="Use existing IV-GICP results (don't re-run)")
    args = parser.parse_args()

    seqs = [args.seq] if args.seq else list(SEQUENCES.keys())
    all_results = {}

    for seq_name in seqs:
        cfg = SEQUENCES[seq_name]
        print(f"\n{'='*60}")
        print(f"  {seq_name}  ({cfg['year']})")
        print(f"{'='*60}")

        bag_path = cfg["bag_local"] if args.local else cfg["bag"]
        if not Path(bag_path).exists():
            if args.local:
                print(f"  [SKIP] local bag not ready: {bag_path}")
                continue
            bag_path = cfg["bag"]

        out_dir = OUT_DIR / seq_name.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Load frames once ──
        frames = read_bag(bag_path, cfg["topic"], cfg["parser"])
        timestamps = [f[0] for f in frames]

        results_this = {}

        # ── KISS-ICP ──
        kiss_tum = out_dir / "kiss_icp.txt"
        if not kiss_tum.exists():
            kiss_poses, kiss_ts = run_kiss(frames, voxel_size=cfg["kiss_voxel"])
            save_tum(kiss_poses, kiss_ts, kiss_tum)
        else:
            print(f"  [KISS-ICP] already done, loading {kiss_tum.name}")

        # ── GenZ-ICP ──
        genz_tum = out_dir / "genz_icp.txt"
        if not genz_tum.exists():
            with tempfile.TemporaryDirectory() as tmp:
                genz_poses, genz_ts = run_genz(frames, cfg["genz_params"], Path(tmp))
                if genz_poses is not None:
                    save_tum(genz_poses, timestamps[:len(genz_poses)], genz_tum)
        else:
            print(f"  [GenZ-ICP] already done, loading {genz_tum.name}")

        # ── IV-GICP ──
        iv_tum = out_dir / "iv_gicp.txt"
        if not iv_tum.exists():
            if not args.skip_iv and Path(cfg["iv_traj"]).exists():
                import shutil
                shutil.copy(cfg["iv_traj"], iv_tum)
                print(f"  [IV-GICP] using existing result")
            else:
                iv_poses = run_ivgicp(frames, cfg["iv_params"])
                save_tum(iv_poses, timestamps, iv_tum)
        else:
            print(f"  [IV-GICP] already done, loading {iv_tum.name}")

        # ── Evaluate ──
        gt_file = cfg["gt"]
        if not Path(gt_file).exists():
            print(f"  [SKIP eval] GT not found: {gt_file}")
            continue

        print(f"\n  --- APE vs GT ({Path(gt_file).name}) ---")
        st_iv   = evaluate_ape(str(iv_tum),   gt_file, "IV-GICP")
        st_kiss = evaluate_ape(str(kiss_tum),  gt_file, "KISS-ICP")
        if genz_tum.exists():
            st_genz = evaluate_ape(str(genz_tum), gt_file, "GenZ-ICP")
        else:
            st_genz = None

        results_this = {
            "iv_rmse":   st_iv["rmse"],
            "kiss_rmse": st_kiss["rmse"],
            "genz_rmse": st_genz["rmse"] if st_genz else None,
        }
        all_results[seq_name] = results_this

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  {'Sequence':<22} {'IV-GICP':>10} {'KISS-ICP':>10} {'GenZ-ICP':>10}")
    print(f"  {'-'*52}")
    for seq, r in all_results.items():
        iv   = f"{r['iv_rmse']*100:.2f}cm"   if r.get("iv_rmse")   else "  N/A   "
        ki   = f"{r['kiss_rmse']*100:.2f}cm"  if r.get("kiss_rmse") else "  N/A   "
        gz   = f"{r['genz_rmse']*100:.2f}cm"  if r.get("genz_rmse") else "  N/A   "
        print(f"  {seq:<22} {iv:>10} {ki:>10} {gz:>10}")


if __name__ == "__main__":
    main()

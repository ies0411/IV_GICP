#!/usr/bin/env python3
"""
AMTC Mine Dataset: IV-GICP vs KISS-ICP benchmark.

Bag: /home/km/data/AMTC/00S_data.bag
  /riegl  — sensor_msgs/PointCloud  ~110 Hz, ~1027 pts/msg
              channels: ['reflectance'] in dBm
  /tf     — static robot URDF transforms only

GT: scanPoseEstimates.dat.tar.bz2
  Format: scanID qx qy qz qw tx ty tz
  44 complete scan poses (00S~43S)

Strategy: msgs_per_scan=182 → 44 GT-aligned full-rotation frames but ~29m
  inter-frame displacement (ICP cannot bridge). Instead use smaller
  msgs_per_frame (default=5) → ~1600 frames with GT poses interpolated
  between the 44 scan poses.

Usage:
    uv run python examples/run_amtc_eval.py
    uv run python examples/run_amtc_eval.py --msgs-per-frame 10 --max-frames 400
    uv run python examples/run_amtc_eval.py --msgs-per-frame 182  # original (likely fails)
"""

import argparse
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "examples"))

AMTC_DIR = Path("/home/km/data/AMTC")
AMTC_BAG = AMTC_DIR / "00S_data.bag"
AMTC_GT  = AMTC_DIR / "scanPoseEstimates.dat.tar.bz2"


def load_amtc_gt(gt_path: Path = AMTC_GT):
    """
    Parse scanPoseEstimates.dat → list of 4×4 SE(3) poses, relative to scan 00S.
    Format per line: scanID qx qy qz qw tx ty tz
    """
    with tarfile.open(gt_path, "r:bz2") as tf:
        member = tf.getmembers()[0]
        text = tf.extractfile(member).read().decode()

    poses = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        qx, qy, qz, qw = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz      = float(parts[5]), float(parts[6]), float(parts[7])
        T = np.eye(4)
        T[0, 0] = 1 - 2*(qy**2 + qz**2)
        T[0, 1] = 2*(qx*qy - qw*qz)
        T[0, 2] = 2*(qx*qz + qw*qy)
        T[1, 0] = 2*(qx*qy + qw*qz)
        T[1, 1] = 1 - 2*(qx**2 + qz**2)
        T[1, 2] = 2*(qy*qz - qw*qx)
        T[2, 0] = 2*(qx*qz - qw*qy)
        T[2, 1] = 2*(qy*qz + qw*qx)
        T[2, 2] = 1 - 2*(qx**2 + qy**2)
        T[:3, 3] = [tx, ty, tz]
        poses.append(T)

    # Make relative to first pose
    T0_inv = np.linalg.inv(poses[0])
    return [T0_inv @ p for p in poses]


def _interp_pose(T0: np.ndarray, T1: np.ndarray, t: float) -> np.ndarray:
    """Linearly interpolate SE(3) pose. t in [0, 1]."""
    if t <= 0.0:
        return T0.copy()
    if t >= 1.0:
        return T1.copy()
    trans = (1.0 - t) * T0[:3, 3] + t * T1[:3, 3]
    # Rotation: geodesic interpolation via R_rel = R0^T R1
    R0, R1 = T0[:3, :3], T1[:3, :3]
    R_rel = R0.T @ R1
    # Rodrigues angle-axis
    cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if abs(angle) < 1e-9:
        R_interp = R0
    else:
        skew = (R_rel - R_rel.T) / (2.0 * np.sin(angle))
        axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        a = angle * t
        c, s = np.cos(a), np.sin(a)
        ux, uy, uz = axis
        R_step = np.array([
            [c + ux*ux*(1-c),   ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
            [uy*ux*(1-c)+uz*s,  c + uy*uy*(1-c),  uy*uz*(1-c)-ux*s],
            [uz*ux*(1-c)-uy*s,  uz*uy*(1-c)+ux*s, c + uz*uz*(1-c) ],
        ])
        R_interp = R0 @ R_step
    T = np.eye(4)
    T[:3, :3] = R_interp
    T[:3, 3]  = trans
    return T


def load_amtc(bag_path: Path = AMTC_BAG,
              max_frames: int = None,
              msgs_per_frame: int = 5,
              min_range: float = 0.3,
              max_range: float = 40.0):
    """
    Load AMTC mine bag → (frames, gt_poses).

    msgs_per_frame consecutive /riegl profiles are merged into one ICP frame.
    GT poses are interpolated between the 44 scan poses so every frame has a
    corresponding ground-truth pose.

    Smaller msgs_per_frame → more frames, smaller inter-frame displacement:
      msgs=5  → ~1600 frames, ~0.8m/fr  (recommended)
      msgs=10 → ~800 frames,  ~1.6m/fr
      msgs=182 → 44 frames,   ~29m/fr   (original, ICP will fail)
    """
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import get_types_from_msg, get_typestore, Stores

    gt_all = load_amtc_gt(AMTC_GT)
    n_gt   = len(gt_all)   # 44

    typestore = get_typestore(Stores.ROS1_NOETIC)

    # ── 1. Collect all filtered point messages ──────────────────────────────
    print("[AMTC] Reading bag ...", flush=True)
    raw_pts, raw_ref = [], []
    with Reader(str(bag_path)) as r:
        add = {}
        for c in r.connections:
            add.update(get_types_from_msg(c.msgdef[1], c.msgtype))
        typestore.register(add)
        pc_conns = [c for c in r.connections if c.topic == "/riegl"]
        for conn, t, raw in r.messages(connections=pc_conns):
            msg  = typestore.deserialize_ros1(raw, conn.msgtype)
            pts  = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)
            refl = np.array(msg.channels[0].values, dtype=np.float32)
            r_dist = np.linalg.norm(pts, axis=1)
            mask   = (r_dist > min_range) & (r_dist < max_range) & np.isfinite(r_dist)
            raw_pts.append(pts[mask])
            raw_ref.append(refl[mask])

    total_msgs = len(raw_pts)
    print(f"[AMTC] {total_msgs} /riegl msgs  →  {total_msgs // msgs_per_frame} frames "
          f"at msgs_per_frame={msgs_per_frame}")

    # Diagnostic: per-GT-step displacement
    dists = [np.linalg.norm(gt_all[i+1][:3, 3] - gt_all[i][:3, 3])
             for i in range(n_gt - 1)]
    print(f"[AMTC] GT inter-scan distance: "
          f"min={min(dists):.2f}m  mean={np.mean(dists):.2f}m  max={max(dists):.2f}m")
    expected_per_frame = np.mean(dists) * msgs_per_frame / (total_msgs / (n_gt - 1))
    print(f"[AMTC] Expected displacement/frame ≈ {expected_per_frame:.2f}m")

    # Detect if one frame = one GT scan (msgs_per_frame ≈ total_msgs / n_gt)
    # In that case use exact GT poses directly (stop-and-scan mode).
    msgs_per_gt = total_msgs / max(n_gt - 1, 1)
    use_exact_gt = abs(msgs_per_frame - msgs_per_gt) < msgs_per_gt * 0.5

    if use_exact_gt:
        print(f"[AMTC] Stop-and-scan mode: using exact GT poses per scan "
              f"(msgs_per_frame={msgs_per_frame} ≈ {msgs_per_gt:.0f} msgs/GT scan)")

    # ── 2. Chunk into frames + assign GT ───────────────────────────────────
    frames, gt_poses = [], []
    for i in range(0, total_msgs - msgs_per_frame + 1, msgs_per_frame):
        chunk_pts = np.vstack(raw_pts[i:i + msgs_per_frame]).astype(np.float64)
        chunk_ref = np.concatenate(raw_ref[i:i + msgs_per_frame]).astype(np.float64)
        if len(chunk_pts) < 50:
            continue
        intensity = np.clip((chunk_ref + 30.0) / 30.0, 0.0, 1.0)
        frames.append(np.column_stack([chunk_pts, intensity]))

        if use_exact_gt:
            # One frame per GT scan: assign GT scan index directly
            gt_idx = min(len(frames) - 1, n_gt - 1)
            gt_poses.append(gt_all[gt_idx])
        else:
            # GT: interpolate by position in message sequence
            frac = (i + msgs_per_frame * 0.5) / total_msgs * (n_gt - 1)
            gt_lo = min(int(frac), n_gt - 2)
            gt_hi = gt_lo + 1
            gt_poses.append(_interp_pose(gt_all[gt_lo], gt_all[gt_hi], frac - gt_lo))

        if max_frames and len(frames) >= max_frames:
            break

    print(f"[AMTC] Loaded {len(frames)} frames  "
          f"(avg {np.mean([len(f) for f in frames]):.0f} pts/frame)  "
          f"GT: {len(gt_poses)} poses")
    return frames, gt_poses


def run_iv_gicp(frames, poses_gt, label="IV-GICP", alpha=0.3, device="cpu",
                gt_init: bool = False):
    """
    gt_init=False : blind odometry (fails for stop-and-scan with large gaps)
    gt_init=True  : GT pose used as initial guess each frame (tests ICP refinement)
    """
    import run_ablation as ra
    from iv_gicp.pipeline import IVGICPPipeline
    pipeline = IVGICPPipeline(
        voxel_size=0.3,
        source_voxel_size=0.15,
        alpha=alpha,
        max_correspondence_distance=2.0,
        initial_threshold=2.0,
        min_motion_th=0.01,
        max_iterations=30,
        max_map_frames=5,
        map_radius=50.0,
        min_range=0.3,
        max_range=40.0,
        device=device,
        auto_alpha=True,
    )
    poses, times_ms = [], []
    for i, f in enumerate(frames):
        init_pose = poses_gt[i] if gt_init else None
        t0 = time.perf_counter()
        result = pipeline.process_frame(
            f[:, :3], f[:, 3], timestamp=float(i),
            **({"init_pose": init_pose} if gt_init and hasattr(pipeline, 'process_frame') else {}),
        )
        times_ms.append((time.perf_counter() - t0) * 1000)
        poses.append(result.pose.copy())
        if (i + 1) % 10 == 0:
            print(f"  [{label}] frame {i+1}/{len(frames)}  "
                  f"{np.mean(times_ms[-10:]):.1f}ms/fr")
    print(f"  [{label}] mean={np.mean(times_ms[1:]):.1f}ms  "
          f"{1000/np.mean(times_ms[1:]):.1f}Hz")
    ate = ra.ate_rmse(poses, poses_gt)
    rpe = ra.rpe_rmse(poses, poses_gt, delta=1)
    return poses, times_ms, ate, rpe


def run_scan_registration(frames, poses_gt, method="iv_gicp", label=None,
                          alpha=0.3, device="cpu"):
    """
    Stop-and-scan evaluation: each pair (scan[i-1], scan[i]) is registered
    independently, initialized with the GT relative transform.

    Metric: per-scan translation error (‖T_est - T_gt‖) and rotation error.
    This avoids error accumulation and is the proper metric for stop-and-scan data.
    """
    import run_ablation as ra

    if label is None:
        label = f"IV-GICP (α={alpha})"

    errs_t, errs_r, times_ms = [], [], []

    if method != "iv_gicp":
        raise ValueError("run_scan_registration only supports method='iv_gicp'.")

    from iv_gicp import IVGICP
    icp = IVGICP(
        min_voxel_size=0.3,
        alpha=alpha,
        max_correspondence_distance=2.0,
        max_iterations=30,
        device=device,
    )

    for i in range(1, len(frames)):
        src  = frames[i]
        tgt  = frames[i - 1]
        T_gt = np.linalg.inv(poses_gt[i - 1]) @ poses_gt[i]

        t0 = time.perf_counter()
        T_est = icp.register(src, tgt, init_pose=T_gt)
        times_ms.append((time.perf_counter() - t0) * 1000)

        # Translation error (m)
        err_t = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
        # Rotation error (deg)
        R_rel = T_est[:3, :3].T @ T_gt[:3, :3]
        cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        err_r = np.degrees(np.arccos(cos_a))
        errs_t.append(err_t)
        errs_r.append(err_r)

        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            print(f"  [{label}] scan {i+1}/{len(frames)}  "
                  f"{np.mean(times_ms[-10:]):.0f}ms  "
                  f"t_err={err_t:.3f}m  r_err={err_r:.2f}°")

    errs_t = np.array(errs_t)
    errs_r = np.array(errs_r)
    print(f"  [{label}] mean={np.mean(times_ms):.0f}ms  "
          f"t_err: mean={np.mean(errs_t):.4f}m median={np.median(errs_t):.4f}m  "
          f"r_err: mean={np.mean(errs_r):.3f}°")
    return errs_t, errs_r, np.array(times_ms)


def run_kiss_icp(frames, poses_gt, voxel_size=0.3):
    import run_ablation as ra
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig()
    cfg.data.max_range = 40.0
    cfg.data.min_range = 0.3
    cfg.data.deskew = False
    cfg.mapping.voxel_size = voxel_size
    od = KissICP(config=cfg)
    poses, times_ms = [], []
    for i, f in enumerate(frames):
        pts = f[:, :3].astype(np.float64)
        t0 = time.perf_counter()
        od.register_frame(pts, np.full(len(pts), float(i)))
        times_ms.append((time.perf_counter() - t0) * 1000)
        poses.append(od.last_pose.copy())
        if (i + 1) % 10 == 0:
            print(f"  [KISS-ICP] frame {i+1}/{len(frames)}  "
                  f"{np.mean(times_ms[-10:]):.1f}ms/fr")
    print(f"  [KISS-ICP] mean={np.mean(times_ms[1:]):.1f}ms  "
          f"{1000/np.mean(times_ms[1:]):.1f}Hz")
    ate = ra.ate_rmse(poses, poses_gt)
    rpe = ra.rpe_rmse(poses, poses_gt, delta=1)
    return poses, times_ms, ate, rpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag",            default=str(AMTC_BAG))
    ap.add_argument("--max-frames",     type=int,   default=None)
    ap.add_argument("--msgs-per-frame", type=int,   default=182)
    ap.add_argument("--alpha",          type=float, default=0.3)
    ap.add_argument("--device",         default="cpu")
    ap.add_argument("--skip-kiss",      action="store_true")
    args = ap.parse_args()

    frames, poses_gt = load_amtc(Path(args.bag), args.max_frames, args.msgs_per_frame)
    if len(frames) < 3:
        print("Too few frames loaded.")
        return

    N = len(frames)
    print(f"\n{'='*65}")
    print(f"AMTC Mine  {N} frames  msgs/frame={args.msgs_per_frame}")
    print(f"{'='*65}\n")

    # ── 1. Blind odometry (IVGICPPipeline vs KISS-ICP) ─────────────────────
    print("── Blind odometry ──")
    iv_poses, iv_times, iv_ate, iv_rpe = run_iv_gicp(
        frames, poses_gt, alpha=args.alpha, device=args.device)

    kiss_ate = kiss_rpe = float('nan')
    kiss_times = [1.0]
    if not args.skip_kiss:
        _, kiss_times, kiss_ate, kiss_rpe = run_kiss_icp(frames, poses_gt)

    print(f"\n{'='*65}")
    print(f"  {'Method':<22} {'ATE(m)':>8}  {'RPE(m)':>8}  {'ms/fr':>8}  {'Hz':>6}")
    print(f"  {'-'*60}")
    print(f"  {'IV-GICP':<22} {iv_ate:>8.4f}  {iv_rpe:>8.4f}  "
          f"{np.mean(iv_times[1:]):>8.1f}  {1000/np.mean(iv_times[1:]):>6.1f}")
    if not args.skip_kiss:
        print(f"  {'KISS-ICP':<22} {kiss_ate:>8.4f}  {kiss_rpe:>8.4f}  "
              f"{np.mean(kiss_times[1:]):>8.1f}  {1000/np.mean(kiss_times[1:]):>6.1f}")
    print(f"{'='*65}")

    # ── 2. GT-initialized scan registration (proper metric) ────────────────
    print(f"\n── GT-initialized scan registration (per-pair accuracy) ──")
    iv_errs_t, iv_errs_r, iv_sr_times = run_scan_registration(
        frames, poses_gt, method="iv_gicp", alpha=args.alpha, device=args.device)

    def fmt_row(label, et, er, tm):
        return (f"  {label:<28} {np.mean(et):>8.4f}  {np.median(et):>7.4f}  "
                f"{np.max(et):>7.4f}  {np.mean(er):>5.2f}°  {np.mean(tm):>7.0f}ms")

    print(f"\n{'='*75}")
    print(f"  {'Method':<28} {'t_mean':>8}  {'t_med':>7}  {'t_max':>7}  "
          f"{'r_mean':>6}  {'ms/scan':>8}")
    print(f"  {'-'*73}")
    print(fmt_row(f"IV-GICP (α={args.alpha})", iv_errs_t, iv_errs_r, iv_sr_times))
    print(f"{'='*75}")


if __name__ == "__main__":
    main()

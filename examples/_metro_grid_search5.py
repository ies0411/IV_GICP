"""
Metro Tunnel Grid Search - Phase 5: 500fr Stability Search
============================================================
200fr에서 좋던 config가 500fr에서 폭발하는 이유:
- sv=0.25, mc=1.0, itr=12 조합이 seq1 fr200-500에서 path 폭발
- 이는 voxel map이 커질수록 source resolution이 너무 조밀해서 오버피팅 가능성
- OR: itr=12는 충분하지 않아서 늦은 프레임에서 수렴 실패

Strategy:
1) seq1 500fr에서 안정적인 config 탐색
2) mc=0.8 (seq2 best) vs mc=1.0 (seq3 best) trade-off
3) 200fr 결과가 500fr에도 일관되는 config 찾기

Usage:
    uv run python examples/_metro_grid_search5.py --device cpu
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

GEODE_ROOT = Path("/home/km/deepai_dev_data/GEODE")
LIDAR_TOPIC = "/livox/lidar"


def parse_livox_frame(raw: bytes, max_range: float = 60.0):
    data = bytes(raw)
    secs, nsecs = struct.unpack_from('<II', data, 4)
    fid_len = struct.unpack_from('<I', data, 12)[0]
    off = 16 + fid_len
    off += 8 + 4 + 1 + 3
    arr_len = struct.unpack_from('<I', data, off)[0]
    off += 4
    arr = np.frombuffer(data[off:off + 19 * arr_len], dtype=np.uint8).reshape(arr_len, 19)
    x    = arr[:,  4: 8].view(np.float32).reshape(-1)
    y    = arr[:,  8:12].view(np.float32).reshape(-1)
    z    = arr[:, 12:16].view(np.float32).reshape(-1)
    refl = arr[:, 16].astype(np.float64) / 255.0
    pts  = np.stack([x, y, z, refl], axis=1).astype(np.float64)
    r    = np.linalg.norm(pts[:, :3], axis=1)
    valid = np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < max_range)
    return secs * 1e9 + nsecs, pts[valid]


def read_metro_frames(bag_path, max_frames=None, max_range=60.0):
    from rosbags.rosbag1 import Reader
    frames = []
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        total = conns[0].msgcount if conns else 0
        for conn, ts_ns, raw in bag.messages(connections=conns):
            try:
                t_ns, pts = parse_livox_frame(raw, max_range=max_range)
                if pts is not None and len(pts) > 100:
                    frames.append((t_ns / 1e9, pts))
                    if len(frames) % 200 == 0:
                        print(f"  {len(frames)}/{total}...", end="\r")
                    if max_frames and len(frames) >= max_frames:
                        break
            except Exception:
                continue
    print(f"\n  Loaded {len(frames)} frames  avg {np.mean([len(f[1]) for f in frames]):.0f} pts/frame")
    return frames


def load_gt(gt_path):
    from scipy.spatial.transform import Rotation
    data = np.loadtxt(gt_path)
    ts, txyz, quats = data[:, 0], data[:, 1:4], data[:, 4:8]
    T0_inv, poses = None, []
    for i in range(len(ts)):
        q = quats[i]
        R = Rotation.from_quat(q).as_matrix() if np.linalg.norm(q) > 1e-6 else np.eye(3)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = txyz[i]
        if T0_inv is None: T0_inv = np.linalg.inv(T)
        poses.append(T0_inv @ T)
    return ts, np.stack(poses)


def interpolate_gt(gt_times, gt_poses, lidar_times):
    idx  = np.searchsorted(gt_times, lidar_times)
    idx  = np.clip(idx, 0, len(gt_times) - 1)
    prev = np.clip(idx - 1, 0, len(gt_times) - 1)
    best = np.where(np.abs(gt_times[prev]-lidar_times) < np.abs(gt_times[idx]-lidar_times), prev, idx)
    return gt_poses[best]


def save_tum(poses, timestamps, path):
    from scipy.spatial.transform import Rotation
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write(f"{ts:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def traj_stats(poses):
    d = [np.linalg.norm(poses[i+1][:3,3]-poses[i][:3,3]) for i in range(len(poses)-1)]
    return float(np.sum(d)), float(np.linalg.norm(poses[-1][:3,3]-poses[0][:3,3]))


def compose_poses(rel_list):
    poses = [np.eye(4)]
    for T in rel_list: poses.append(poses[-1] @ T)
    return poses


def compute_ate(gt_tum, pred_tum):
    try:
        from evo.tools import file_interface
        from evo.core import sync, metrics
        ref = file_interface.read_tum_trajectory_file(str(gt_tum))
        est = file_interface.read_tum_trajectory_file(str(pred_tum))
        ref, est = sync.associate_trajectories(ref, est)
        est.align(ref, correct_scale=False)
        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((ref, est))
        return float(ape.get_statistic(metrics.StatisticsType.rmse))
    except Exception:
        return None


def run_config(frames, timestamps, cfg, device, tmp_dir, tag):
    from iv_gicp.pipeline import IVGICPPipeline
    pipeline = IVGICPPipeline(
        voxel_size=cfg["voxel_size"],
        source_voxel_size=cfg["source_voxel_size"],
        alpha=cfg["alpha"],
        max_correspondence_distance=cfg["max_correspondence_distance"],
        initial_threshold=cfg.get("initial_threshold", 1.5),
        min_motion_th=cfg.get("min_motion_th", 0.5),
        max_map_frames=cfg.get("max_map_frames", 200),
        max_iterations=cfg.get("max_iterations", 20),
        map_radius=cfg.get("map_radius", 60.0),
        auto_alpha=False, auto_alpha_from_intensity=False,
        source_drop_small_voxels=False, source_max_output_features=0,
        source_min_feature_score=0.0, max_source_points=0,
        device=device,
    )
    abs_poses = [np.eye(4)]; times = []
    for ts, pts in frames:
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        times.append((time.perf_counter() - t0) * 1000)
        abs_poses.append(result.pose.copy())
    rel = [np.linalg.inv(abs_poses[i-1]) @ abs_poses[i] for i in range(1, len(abs_poses))]
    poses = compose_poses(rel)
    tum = Path(tmp_dir) / f"{tag}.tum"
    save_tum(poses, timestamps, tum)
    path_len, end_disp = traj_stats(poses)
    return path_len, end_disp, float(np.mean(times)), tum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    tmp_dir = Path(__file__).parent.parent / "results" / "geode" / "_gridsearch5"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Phase 5: Focus on 500fr stability
    # Key issue: sv=0.25, itr=12, mc<1.5 causes path explosion in seq1 500fr
    # Hypothesis: lower itr + smaller mc → early-frame convergence ok, but
    #             as trajectory grows longer, small ICP radius misses correspondences

    configs = {
        # Baseline (original paper config)
        "orig_a0.5_sv0.2_mc1.5_itr20_mr60": dict(
            voxel_size=0.5, source_voxel_size=0.2, alpha=0.5,
            max_correspondence_distance=1.5, initial_threshold=1.5,
            min_motion_th=0.5, max_map_frames=200, max_iterations=20, map_radius=60.0),

        # Phase 3 global best (200fr)
        "best200_a0_sv0.25_mc1.0_itr12_mr60": dict(
            voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
            max_correspondence_distance=1.0, initial_threshold=1.5,
            min_motion_th=0.5, max_map_frames=200, max_iterations=12, map_radius=60.0),

        # mc=0.8 (best on seq2 500fr, nearly ties KISS)
        "mc0.8_sv0.25_itr12": dict(
            voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
            max_correspondence_distance=0.8, initial_threshold=1.5,
            min_motion_th=0.5, max_map_frames=200, max_iterations=12, map_radius=60.0),

        # itr=20 helps convergence for longer sequences?
        "sv0.25_mc1.0_itr20_mr60": dict(
            voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
            max_correspondence_distance=1.0, initial_threshold=1.5,
            min_motion_th=0.5, max_map_frames=200, max_iterations=20, map_radius=60.0),

        # alpha=0.0, mc=1.5 (from phase1 seq1 best: baseline_alpha0_mr60)
        "a0_sv0.2_mc1.5_itr20_mr60": dict(
            voxel_size=0.5, source_voxel_size=0.2, alpha=0.0,
            max_correspondence_distance=1.5, initial_threshold=1.5,
            min_motion_th=0.5, max_map_frames=200, max_iterations=20, map_radius=60.0),

        # Phase1 seq1 absolute winner
        "a0_sv0.2_mc1.5_itr20_mrNone": dict(
            voxel_size=0.5, source_voxel_size=0.2, alpha=0.0,
            max_correspondence_distance=1.5, initial_threshold=1.5,
            min_motion_th=0.5, max_map_frames=200, max_iterations=20, map_radius=None),

        # NEW: smaller source voxel helps but need larger mc to compensate
        "sv0.25_mc1.5_itr20_mr60": dict(
            voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
            max_correspondence_distance=1.5, initial_threshold=1.5,
            min_motion_th=0.5, max_map_frames=200, max_iterations=20, map_radius=60.0),

        # sv=0.25, mc=1.0, itr=20 (fix convergence)
        "sv0.25_mc1.0_itr20_mrNone": dict(
            voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
            max_correspondence_distance=1.0, initial_threshold=1.5,
            min_motion_th=0.5, max_map_frames=200, max_iterations=20, map_radius=None),

        # Large mc, large itr (aggressive convergence)
        "sv0.25_mc2.0_itr20": dict(
            voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
            max_correspondence_distance=2.0, initial_threshold=1.5,
            min_motion_th=0.5, max_map_frames=200, max_iterations=20, map_radius=60.0),

        # mc=0.8 + itr=20
        "mc0.8_sv0.25_itr20": dict(
            voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
            max_correspondence_distance=0.8, initial_threshold=1.5,
            min_motion_th=0.5, max_map_frames=200, max_iterations=20, map_radius=60.0),

        # Try wider map (more context for long sequences)
        "sv0.25_mc1.0_itr12_mr80": dict(
            voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
            max_correspondence_distance=1.0, initial_threshold=1.5,
            min_motion_th=0.5, max_map_frames=200, max_iterations=12, map_radius=80.0),

        # Larger map_frames to retain more history
        "sv0.25_mc1.0_itr12_mf500": dict(
            voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
            max_correspondence_distance=1.0, initial_threshold=1.5,
            min_motion_th=0.5, max_map_frames=500, max_iterations=12, map_radius=60.0),
    }

    all_results = {}

    for seq_num in ["1", "2", "3"]:
        seq_bag = f"Shield_tunnel{seq_num}_gamma"
        bag_path = GEODE_ROOT/"sensor_data"/"Metro_tunnel"/seq_bag/f"{seq_bag}.bag"
        gt_path  = GEODE_ROOT/"groundtruth"/"metro_tunnel"/f"Shield_tunnel{seq_num}.txt"
        seq_tmp  = tmp_dir / f"seq{seq_num}"
        seq_tmp.mkdir(parents=True, exist_ok=True)

        print(f"\n{'═'*72}")
        print(f"  Seq {seq_num}  ({args.max_frames}fr)")
        frames = read_metro_frames(bag_path, max_frames=args.max_frames)
        n = len(frames)
        timestamps = [f[0] for f in frames]

        gt_times, gt_poses = load_gt(gt_path)
        gt_at_lidar = interpolate_gt(gt_times, gt_poses, np.array(timestamps))
        gt_tum = seq_tmp / "gt.tum"
        save_tum(gt_at_lidar, timestamps, gt_tum)
        gt_pl, gt_ed = traj_stats(gt_at_lidar)
        print(f"  GT: path={gt_pl:.1f}m  end={gt_ed:.1f}m")

        # KISS
        from kiss_icp.kiss_icp import KissICP
        from kiss_icp.config import KISSConfig
        kc = KISSConfig()
        kc.data.deskew = False; kc.data.max_range = 60.0; kc.data.min_range = 0.5
        kc.mapping.voxel_size = 0.8
        od = KissICP(config=kc)
        kiss_poses = [np.eye(4)]; kiss_times = []
        for ts, pts in frames:
            t0 = time.perf_counter()
            od.register_frame(pts[:, :3].astype(np.float64), np.full(len(pts), ts))
            kiss_times.append((time.perf_counter() - t0) * 1000)
            kiss_poses.append(od.last_pose.copy())
        kiss_tum = seq_tmp / "kiss.tum"
        save_tum(kiss_poses, timestamps, kiss_tum)
        kiss_pl2, kiss_ed2 = traj_stats(kiss_poses)
        kiss_ate = compute_ate(gt_tum, kiss_tum)
        print(f"  KISS: path={kiss_pl2:.1f}m  end={kiss_ed2:.1f}m  ATE={kiss_ate:.3f}m  "
              f"({np.mean(kiss_times):.1f}ms)")

        results = {"KISS": {"ate": kiss_ate, "path_len": kiss_pl2, "mean_ms": float(np.mean(kiss_times))}}

        print(f"\n  {'Tag':<44} {'Path':>7} {'End':>7} {'ATE':>9} {'ms':>6} {'vs KISS':>8}")
        print(f"  {'-'*80}")

        for tag, cfg_i in configs.items():
            try:
                pl, ed, ms, tum = run_config(frames, timestamps, cfg_i, args.device, seq_tmp, tag)
                ate = compute_ate(gt_tum, tum)
                pct = f"{(ate - kiss_ate)/kiss_ate*100:+.1f}%" if ate and kiss_ate else "N/A"
                ate_s = f"{ate:.3f}m" if ate else "N/A"
                marker = " ✓" if ate and kiss_ate and ate < kiss_ate else ""
                print(f"  {tag:<44} {pl:>7.1f} {ed:>7.1f} {ate_s:>9} {ms:>6.1f} {pct:>8}{marker}")
                results[tag] = {"ate": ate, "path_len": pl, "end_disp": ed, "mean_ms": ms}
            except Exception as e:
                print(f"  {tag:<44} ERROR: {e}")

        all_results[f"seq{seq_num}"] = {
            "gt_path_m": gt_pl, "n_frames": n, "results": results}

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PHASE 5 SUMMARY  {args.max_frames}fr/seq")
    print(f"{'='*80}")

    improvements = {}
    for tag in configs.keys():
        ates, kiss_ates = [], []
        for sk in all_results.keys():
            res = all_results[sk]["results"]
            if tag in res and res[tag].get("ate"):
                ates.append(res[tag]["ate"])
                kiss_ates.append(res["KISS"]["ate"])
        if ates and kiss_ates:
            improvements[tag] = (
                np.mean([(a-k)/k*100 for a, k in zip(ates, kiss_ates)]),
                np.mean(ates), ates
            )

    kiss_avg = np.mean([all_results[sk]["results"]["KISS"]["ate"] for sk in all_results])
    kiss_ates = [all_results[sk]["results"]["KISS"]["ate"] for sk in all_results]
    print(f"  KISS avg={kiss_avg:.3f}m  [{' '.join(f'{a:.3f}' for a in kiss_ates)}]")
    print()

    for tag, (pct, avg, ates) in sorted(improvements.items(), key=lambda x: x[1][0]):
        paths = [all_results[sk]["results"].get(tag, {}).get("path_len", float("nan"))
                 for sk in all_results]
        ates_s = "  ".join([f"{a:.3f}" for a in ates])
        paths_s = "  ".join([f"{p:.0f}" for p in paths])
        marker = " <-- BEST" if tag == sorted(improvements.items(), key=lambda x: x[1][0])[0][0] else ""
        print(f"  {tag:<46} avg={avg:.3f}m  {pct:+.1f}%  ATE=[{ates_s}]  path=[{paths_s}]{marker}")

    out_json = tmp_dir / f"phase5_summary_{args.max_frames}fr.json"
    with open(out_json, "w") as f:
        json.dump({"results": all_results}, f, indent=2, default=str)
    print(f"\n  Results → {out_json}")


if __name__ == "__main__":
    main()

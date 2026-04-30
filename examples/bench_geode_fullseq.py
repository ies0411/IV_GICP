#!/usr/bin/env python3
"""
GEODE Urban Tunnel — FULL-sequence benchmark with auto-gate sweep.

Runs on Tunnel01 (2857 fr), Tunnel02 (3425 fr), Tunnel03 (3350 fr).

Configs per sequence:
  KISS-ICP (baseline)
  IV-GICP C1_off               — alpha=0.0, use_fim_weight=False
  IV-GICP C1_on   (paper cfg)  — alpha=0.0, use_fim_weight=True
  IV-GICP gate_0.005           — auto-gate at 0.005
  IV-GICP gate_0.020           — auto-gate at 0.020
  IV-GICP gate_0.050           — auto-gate at 0.050

Outputs results/geode/fullseq/<seq>/{method}.tum + summary.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))

GEODE_ROOT = Path("/home/km/deepai_dev_data/GEODE")
LIDAR_TOPIC = "/velodyne_points"
OUT_ROOT = Path(__file__).parent.parent / "results" / "geode" / "fullseq"


def read_geode_frames(bag_path: Path, max_frames=None):
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    frames = []
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        for conn, ts_ns, raw in bag.messages(connections=conns):
            m = ts.deserialize_ros1(raw, conn.msgtype)
            d = np.frombuffer(bytes(m.data), dtype=np.uint8).reshape(-1, m.point_step)
            x = d[:, 0:4].view(np.float32).reshape(-1)
            y = d[:, 4:8].view(np.float32).reshape(-1)
            z = d[:, 8:12].view(np.float32).reshape(-1)
            it = d[:, 12:16].view(np.float32).reshape(-1)
            pts = np.stack([x, y, z, it], axis=1).astype(np.float64)
            r = np.linalg.norm(pts[:, :3], axis=1)
            pts = pts[np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < 80.0)]
            if len(pts) > 100:
                frames.append((ts_ns / 1e9, pts))
            if max_frames and len(frames) >= max_frames:
                break
    return frames


def load_geode_gt(gt_path: Path):
    data = np.loadtxt(gt_path)
    T0_inv = None
    out = []
    for row in data:
        R = Rotation.from_quat(row[4:8]).as_matrix()
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = row[1:4]
        if T0_inv is None:
            T0_inv = np.linalg.inv(T)
        out.append(T0_inv @ T)
    return data[:, 0], np.stack(out)


def interp_gt(gt_ts, gt_poses, ts):
    idx = np.searchsorted(gt_ts, ts)
    idx = np.clip(idx, 0, len(gt_ts) - 1)
    prev = np.clip(idx - 1, 0, len(gt_ts) - 1)
    before = np.abs(gt_ts[prev] - ts)
    after = np.abs(gt_ts[idx] - ts)
    best = np.where(before < after, prev, idx)
    return gt_poses[best]


def save_tum(poses, timestamps, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write(f"{ts:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def ate_rmse(gt_tum: Path, pred_tum: Path):
    from evo.tools import file_interface
    from evo.core import sync, metrics
    tref = file_interface.read_tum_trajectory_file(str(gt_tum))
    test = file_interface.read_tum_trajectory_file(str(pred_tum))
    tref, test = sync.associate_trajectories(tref, test)
    test.align(tref, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((tref, test))
    return float(ape.get_statistic(metrics.StatisticsType.rmse))


def run_kiss(frames):
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig()
    cfg.data.deskew = False
    cfg.data.max_range = 80.0
    cfg.data.min_range = 0.5
    cfg.mapping.voxel_size = 0.5
    od = KissICP(config=cfg)
    poses = [np.eye(4)]
    times = []
    for i, (ts, pts) in enumerate(frames):
        t0 = time.perf_counter()
        od.register_frame(pts[:, :3].astype(np.float64), np.full(len(pts), ts))
        times.append((time.perf_counter() - t0) * 1000)
        poses.append(od.last_pose.copy())
        if i % 200 == 0:
            print(f"  KISS  {i}/{len(frames)}  {np.mean(times[-20:]):.0f}ms", end="\r")
    print(f"  KISS  done  mean={np.mean(times):.1f}ms   ")
    return poses, times


def run_iv(frames, use_fim_weight=True, fim_auto_gate=0.0, label=""):
    from iv_gicp.pipeline import IVGICPPipeline
    p = IVGICPPipeline(
        voxel_size=0.5,
        source_voxel_size=0.25,
        alpha=0.0,
        max_correspondence_distance=2.0,
        initial_threshold=1.5,
        min_motion_th=0.5,
        max_iterations=12,
        max_map_frames=500,
        map_radius=80.0,
        use_fim_weight=use_fim_weight,
        fim_auto_gate=fim_auto_gate,
        auto_alpha=False,
        auto_alpha_from_intensity=False,
        source_drop_small_voxels=False,
        source_max_output_features=0,
        source_min_feature_score=0.0,
        max_source_points=0,
        device="cpu",
    )
    poses = [np.eye(4)]
    times = []
    reg_t = []
    map_t = []
    for i, (ts, pts) in enumerate(frames):
        t0 = time.perf_counter()
        r = p.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        times.append((time.perf_counter() - t0) * 1000)
        reg_t.append(r.reg_ms)
        map_t.append(r.map_ms)
        poses.append(r.pose.copy())
        if i % 200 == 0:
            print(f"  {label}  {i}/{len(frames)}  {np.mean(times[-20:]):.0f}ms", end="\r")
    print(f"  {label}  done  mean={np.mean(times):.1f}ms  reg={np.mean(reg_t):.1f} map={np.mean(map_t):.1f}  ")
    return poses, times, reg_t, map_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["01", "02", "03"])
    ap.add_argument("--max-frames", type=int, default=None, help="None = full seq")
    ap.add_argument("--skip-kiss", action="store_true")
    ap.add_argument("--configs", nargs="+", default=["c1_off", "c1_on", "g005", "g020", "g050"],
                    help="IV configs to run")
    args = ap.parse_args()

    config_map = {
        "c1_off": dict(use_fim_weight=False, fim_auto_gate=0.0),
        "c1_on":  dict(use_fim_weight=True,  fim_auto_gate=0.0),
        "g005":   dict(use_fim_weight=True,  fim_auto_gate=0.005),
        "g020":   dict(use_fim_weight=True,  fim_auto_gate=0.020),
        "g050":   dict(use_fim_weight=True,  fim_auto_gate=0.050),
    }

    all_results = {}
    for seq in args.seqs:
        seq_name = f"Urban_Tunnel{seq}"
        bag = GEODE_ROOT / "sensor_data" / "Urban_tunnel" / seq_name / f"{seq_name}.bag"
        gt  = GEODE_ROOT / "groundtruth" / "Urban_tunnel" / f"{seq_name}.txt"
        out = OUT_ROOT / seq_name
        out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}\n[{seq_name}]  loading {bag.name}")
        t0 = time.perf_counter()
        frames = read_geode_frames(bag, max_frames=args.max_frames)
        print(f"  loaded {len(frames)} frames in {time.perf_counter()-t0:.1f}s")
        timestamps = [f[0] for f in frames]
        gt_ts, gt_poses = load_geode_gt(gt)
        gt_at_lidar = interp_gt(gt_ts, gt_poses, np.array(timestamps))
        save_tum(gt_at_lidar, timestamps, out / "gt.tum")

        seq_res = {}

        # KISS
        if not args.skip_kiss:
            kp, kt = run_kiss(frames)
            save_tum(kp[1:], timestamps, out / "kiss.tum")
            a = ate_rmse(out / "gt.tum", out / "kiss.tum")
            seq_res["kiss"] = {"ate": a, "ms_mean": float(np.mean(kt)), "n_frames": len(frames)}
            print(f"  KISS  ATE={a:.4f}m")

        # IV variants
        for cfg_name in args.configs:
            cfg = config_map[cfg_name]
            label = f"IV-{cfg_name}"
            ip, it, regt, mapt = run_iv(frames, label=label, **cfg)
            save_tum(ip[1:], timestamps, out / f"iv_{cfg_name}.tum")
            a = ate_rmse(out / "gt.tum", out / f"iv_{cfg_name}.tum")
            seq_res[f"iv_{cfg_name}"] = {
                "ate": a, "ms_mean": float(np.mean(it)),
                "reg_ms_mean": float(np.mean(regt)), "map_ms_mean": float(np.mean(mapt)),
                "n_frames": len(frames),
                **cfg,
            }
            print(f"  IV-{cfg_name}  ATE={a:.4f}m")

        all_results[seq_name] = seq_res
        (out / "summary.json").write_text(json.dumps(seq_res, indent=2))

    # Final report
    print(f"\n{'='*70}\nSUMMARY — GEODE Urban Tunnel Full-Sequence")
    print(f"{'Seq':<16} {'n_fr':>5}   {'kiss':>8}  {'c1_off':>8}  {'c1_on':>8}  {'g005':>8}  {'g020':>8}  {'g050':>8}")
    for seq_name, r in all_results.items():
        n = r.get("kiss", r.get("iv_c1_on", {})).get("n_frames", 0)
        row = f"{seq_name:<16} {n:>5}"
        for k in ["kiss", "iv_c1_off", "iv_c1_on", "iv_g005", "iv_g020", "iv_g050"]:
            v = r.get(k, {}).get("ate")
            row += f"   {v:>6.3f}m" if v is not None else f"   {'  -   ':>8}"
        print(row)

    (OUT_ROOT / "summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {OUT_ROOT}/summary.json")


if __name__ == "__main__":
    main()

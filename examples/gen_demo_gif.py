#!/usr/bin/env python3
"""Generate a top-down 3D map + trajectory GIF for paper supplementary video.

Style reference: docs/paper_material/livox-demo.gif (top-down colored point
cloud with trajectory line). Targets 640x360, ~100 frames, intensity/height
colormap, black background.

Supported sequences:
  subt     → uses existing poses from results/fullseq_logs/subt_Final_UGV1_poses_minth0.5.npz
             Raw frames streamed from SubT Final_UGV1 rosbag zip.
  geode02  → runs IV-GICP on GEODE Urban_Tunnel02 (flagship -86% result) and
             renders IV vs KISS trajectories overlaid.

Usage:
  uv run python examples/gen_demo_gif.py --seq subt    --keyframes 120
  uv run python examples/gen_demo_gif.py --seq geode02 --keyframes 120
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

OUT_DIR = Path(__file__).parent.parent / "docs" / "paper_material"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Voxel downsample ──────────────────────────────────────────────────────────

def voxel_downsample(pts, vox):
    if len(pts) == 0:
        return pts
    keys = np.floor(pts[:, :3] / vox).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


# ── Renderer ──────────────────────────────────────────────────────────────────

def render_frame(map_pts, traj_xy, traj_kiss_xy,
                 view_bounds, current_xy,
                 size=(640, 360), dpi=100,
                 color_mode="height", ptsize=0.5,
                 title=None, traj_genz_xy=None,
                 iv_label="IV-GICP", kiss_label="KISS-ICP",
                 genz_label="GenZ-ICP"):
    """Render a single frame → (H,W,3) uint8.

    map_pts: (N,4) xyz+intensity in world frame.
    traj_xy: (T,2) IV-GICP trajectory up to current frame.
    traj_kiss_xy: (T,2) optional baseline trajectory.
    view_bounds: (xmin, xmax, ymin, ymax).
    current_xy: (x,y) pose marker.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    figsize = (size[0] / dpi, size[1] / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    if len(map_pts) > 0:
        if color_mode == "height":
            c = map_pts[:, 2]
            cmin = np.percentile(c, 2)
            cmax = np.percentile(c, 98)
        elif color_mode == "intensity":
            c = map_pts[:, 3]
            cmin = np.percentile(c, 2)
            cmax = np.percentile(c, 98)
        else:
            c = np.linalg.norm(map_pts[:, :2] - current_xy, axis=1)
            cmin, cmax = 0, c.max() + 1e-6
        cmap = LinearSegmentedColormap.from_list(
            "fire", ["#1a0033", "#aa1a4a", "#ff6e1a", "#ffcf3a", "#ffffcc"])
        ax.scatter(map_pts[:, 0], map_pts[:, 1], c=c, cmap=cmap,
                   vmin=cmin, vmax=cmax, s=ptsize, linewidths=0, alpha=0.85)

    # Trajectories
    if traj_kiss_xy is not None and len(traj_kiss_xy) > 1:
        ax.plot(traj_kiss_xy[:, 0], traj_kiss_xy[:, 1],
                color="#ff4040", lw=1.6, alpha=0.9, label=kiss_label)
    if traj_genz_xy is not None and len(traj_genz_xy) > 1:
        ax.plot(traj_genz_xy[:, 0], traj_genz_xy[:, 1],
                color="#3cdb6a", lw=1.6, alpha=0.9, label=genz_label)
    if len(traj_xy) > 1:
        ax.plot(traj_xy[:, 0], traj_xy[:, 1],
                color="#3ac5ff", lw=2.0, alpha=0.97, label=iv_label)

    # Current pose marker
    ax.plot(current_xy[0], current_xy[1], marker="o",
            ms=7, mfc="#ffffff", mec="#3ac5ff", mew=1.2, zorder=10)

    ax.set_xlim(view_bounds[0], view_bounds[1])
    ax.set_ylim(view_bounds[2], view_bounds[3])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    if title:
        ax.text(0.015, 0.965, title, transform=ax.transAxes,
                color="#eeeeee", fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="#000000cc", ec="#555555", lw=0.5))
    if traj_kiss_xy is not None or traj_genz_xy is not None:
        ax.legend(loc="lower right", fontsize=7, frameon=True,
                  facecolor="#000000cc", edgecolor="#555555", labelcolor="white")

    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


# ── SubT sequence ─────────────────────────────────────────────────────────────

def _subt_stream_keyframes(rosbag_zip, keep_indices):
    """Yield (i, ts_ns, cloud) only for frames in keep_indices.
    Skips VLP-16 decode for skipped frames — massively faster than full decode.
    """
    import os, tempfile, zipfile
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import get_types_from_msg, get_typestore, Stores
    from run_subt_eval import decode_vlp16_scan

    typestore = get_typestore(Stores.ROS1_NOETIC)
    add_types = {}
    for t, msgdef in [
        ('velodyne_msgs/msg/VelodynePacket', 'time stamp\nuint8[1206] data\n'),
        ('velodyne_msgs/msg/VelodyneScan',
         'std_msgs/Header header\nvelodyne_msgs/VelodynePacket[] packets\n'),
    ]:
        add_types.update(get_types_from_msg(msgdef, t))
    typestore.register(add_types)

    keep = set(int(i) for i in keep_indices)
    max_i = max(keep) if keep else -1
    i = 0
    with zipfile.ZipFile(rosbag_zip, 'r') as z:
        bag_names = sorted([n for n in z.namelist() if n.endswith('.bag')])
        for bag_name in bag_names:
            bag_data = z.read(bag_name)
            with tempfile.NamedTemporaryFile(suffix='.bag', delete=False) as tf:
                tf.write(bag_data)
                tmp_path = tf.name
            try:
                with Reader(tmp_path) as reader:
                    conns = [c for c in reader.connections
                             if 'velodyne' in c.topic.lower()
                             and c.msgtype == 'velodyne_msgs/msg/VelodyneScan']
                    if not conns:
                        continue
                    for _, ts_ns, rawdata in reader.messages(connections=conns):
                        if i > max_i:
                            return
                        if i in keep:
                            msg = typestore.deserialize_ros1(
                                rawdata, 'velodyne_msgs/msg/VelodyneScan')
                            packets_data = [bytes(p.data) for p in msg.packets]
                            cloud = decode_vlp16_scan(packets_data)
                            if len(cloud) >= 100:
                                yield i, ts_ns, cloud
                        i += 1
            finally:
                os.unlink(tmp_path)


def build_subt(keyframes=120, map_voxel=0.3, render_voxel=0.15,
               max_map_pts=120_000):
    """SubT Final_UGV1: load poses (existing) + raw frames, accumulate map."""
    from run_subt_eval import DATASETS, BASE_DIR
    import os

    pose_npz = Path("results/fullseq_logs/subt_Final_UGV1_poses_minth0.5.npz")
    data = np.load(pose_npz)
    poses = data["poses"]  # (N,4,4)
    n = len(poses)
    step = max(1, n // keyframes)
    keep_idx = list(range(0, n, step))
    print(f"[subt] {n} poses, keeping every {step}-th → {len(keep_idx)} keyframes")

    info = DATASETS["Final_UGV1"]
    rosbag_zip = os.path.join(BASE_DIR, info["rosbag"])

    acc_map = np.empty((0, 4))
    keyframe_poses = []
    map_history = []
    kf_count = 0
    for i, ts_ns, cloud in _subt_stream_keyframes(rosbag_zip, keep_idx):
        T = poses[i]
        pts = cloud[:, :3].astype(np.float32)
        intensity = (cloud[:, 3] if cloud.shape[1] >= 4
                     else np.zeros(len(pts))).astype(np.float32)
        r = np.linalg.norm(pts, axis=1)
        m = (r > 0.5) & (r < 40.0)
        pts, intensity = pts[m], intensity[m]
        pts_w = (T[:3, :3] @ pts.T).T + T[:3, 3]
        combined = np.column_stack([pts_w, intensity])
        combined = voxel_downsample(combined, render_voxel)
        acc_map = np.concatenate([acc_map, combined], axis=0)
        acc_map = voxel_downsample(acc_map, map_voxel)
        if len(acc_map) > max_map_pts:
            idx = np.random.choice(len(acc_map), max_map_pts, replace=False)
            acc_map = acc_map[idx]
        map_history.append(acc_map.copy())
        keyframe_poses.append(T[:3, 3].copy())
        kf_count += 1
        if kf_count % 10 == 0:
            print(f"  [subt] kf {kf_count}/{len(keep_idx)}  map_pts={len(acc_map)}",
                  flush=True)
    print(f"[subt] built {kf_count} keyframes")
    return map_history, np.array(keyframe_poses)


def render_subt_gif(keyframes=120, fps=15):
    import imageio.v2 as imageio
    map_history, traj = build_subt(keyframes=keyframes)
    # View bounds: compute from final map
    final = map_history[-1]
    pad = 10.0
    xmin, xmax = final[:, 0].min() - pad, final[:, 0].max() + pad
    ymin, ymax = final[:, 1].min() - pad, final[:, 1].max() + pad
    # Preserve aspect (640x360 = 16:9)
    w, h = xmax - xmin, ymax - ymin
    target = 640 / 360
    if w / h > target:
        extra = w / target - h
        ymin -= extra / 2; ymax += extra / 2
    else:
        extra = h * target - w
        xmin -= extra / 2; xmax += extra / 2

    frames = []
    for i, (pts, pose) in enumerate(zip(map_history, traj)):
        title = f"IV-GICP | SubT Final_UGV1 | frame {int((i+1)*len(pts)):>5d}"
        # better title: use keyframe index out of total
        title = f"IV-GICP | SubT Final_UGV1 | kf {i+1}/{len(traj)}"
        img = render_frame(pts, traj[:i+1], None,
                           (xmin, xmax, ymin, ymax),
                           pose[:2], color_mode="height",
                           ptsize=0.5, title=title)
        frames.append(img)
        if (i+1) % 20 == 0:
            print(f"  [subt-render] {i+1}/{len(traj)}")
    out = OUT_DIR / "subt_demo.gif"
    imageio.mimsave(str(out), frames, fps=fps, loop=0)
    print(f"\n[subt] saved → {out}  ({len(frames)} frames, {out.stat().st_size/1e6:.1f} MB)")
    return out


# ── GenZ-ICP subprocess helper ────────────────────────────────────────────────

GENZ_RUNNER = Path(__file__).parent.parent / "thirdparty/genz-icp/bin_runner/build/genz_bin_runner"


def run_genz_on_clouds(clouds, voxel_size=0.5, max_range=80.0,
                      min_range=0.5, min_motion_th=0.5,
                      initial_threshold=1.5, planarity_th=0.1):
    """Dump clouds → .bin, run genz_bin_runner, return (N,4,4) poses."""
    import subprocess, tempfile
    from pathlib import Path as _P
    with tempfile.TemporaryDirectory(prefix="genz_demo_") as tmp:
        tp = _P(tmp)
        for i, pts in enumerate(clouds):
            pts[:, :4].astype(np.float32).tofile(tp / f"{i:06d}.bin")
        poses_out = tp / "genz_poses.txt"
        cmd = [str(GENZ_RUNNER), str(tp), str(poses_out),
               "--max-frames", str(len(clouds)),
               "--voxel-size", str(voxel_size),
               "--max-range", str(max_range),
               "--min-range", str(min_range),
               "--min-motion-th", str(min_motion_th),
               "--initial-threshold", str(initial_threshold),
               "--planarity-th", str(planarity_th)]
        print(f"  [genz] running on {len(clouds)} frames...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"genz failed: {res.stderr[-500:]}")
        poses = []
        with open(poses_out) as f:
            for line in f:
                vals = list(map(float, line.split()))
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
    print(f"  [genz] {len(poses)} poses")
    return poses


# ── GEODE 3-way (generic Urban_Tunnel01/02/03) ────────────────────────────────

GEODE_SPECS = {
    "geode01": dict(name="Urban_Tunnel01", iv_ate=1.00, kiss_ate=1.91, genz_ate=1.36),
    "geode02": dict(name="Urban_Tunnel02", iv_ate=1.90, kiss_ate=13.70, genz_ate=10.90),
    "geode03": dict(name="Urban_Tunnel03", iv_ate=4.74, kiss_ate=5.45, genz_ate=5.37),
}


def build_geode_3way(seq_key, keyframes=120, map_voxel=0.4, render_voxel=0.2,
                     max_map_pts=120_000):
    """Run IV + KISS + GenZ on a GEODE Urban_Tunnel sequence; accumulate IV-map."""
    from run_geode_eval import read_geode_frames, GEODE_ROOT
    from iv_gicp.pipeline import IVGICPPipeline
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    spec = GEODE_SPECS[seq_key]
    seq_name = spec["name"]
    bag_path = GEODE_ROOT / "sensor_data" / "Urban_tunnel" / seq_name / f"{seq_name}.bag"
    print(f"[{seq_key}] loading {bag_path}")
    frames = read_geode_frames(bag_path, max_frames=None)
    n = len(frames)
    print(f"[{seq_key}] {n} frames loaded")

    pipeline = IVGICPPipeline(
        voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
        max_correspondence_distance=2.0, initial_threshold=1.5,
        min_motion_th=0.5, max_iterations=12, max_map_frames=500,
        map_radius=80.0, use_fim_weight=True, auto_alpha=False,
        auto_alpha_from_intensity=False, source_drop_small_voxels=False,
        source_max_output_features=0, source_min_feature_score=0.0,
        max_source_points=0, device="cpu",
    )
    cfg = KISSConfig()
    cfg.data.deskew = False; cfg.data.max_range = 80.0; cfg.data.min_range = 0.5
    cfg.mapping.voxel_size = 0.5
    kiss = KissICP(config=cfg)

    iv_poses = [np.eye(4)]
    kiss_poses = [np.eye(4)]
    genz_clouds = []

    step = max(1, n // keyframes)
    keep_idx = set(range(0, n, step))
    acc_map = np.empty((0, 4))
    map_history = []
    iv_traj = []
    kiss_traj = []
    keep_frame_iv_idx = []

    for i, (ts, pts) in enumerate(frames):
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        iv_poses.append(result.pose.copy())
        kiss.register_frame(pts[:, :3].astype(np.float64), np.full(len(pts), ts))
        kiss_poses.append(kiss.last_pose.copy())
        genz_clouds.append(pts[:, :4].astype(np.float32))

        if i in keep_idx or i == n - 1:
            Tw = result.pose
            p = pts[:, :3].astype(np.float32)
            intensity = pts[:, 3].astype(np.float32)
            r = np.linalg.norm(p, axis=1)
            m = (r > 0.5) & (r < 60.0)
            p, intensity = p[m], intensity[m]
            pw = (Tw[:3, :3] @ p.T).T + Tw[:3, 3]
            combined = voxel_downsample(
                np.column_stack([pw, intensity]), render_voxel)
            acc_map = np.concatenate([acc_map, combined], axis=0)
            acc_map = voxel_downsample(acc_map, map_voxel)
            if len(acc_map) > max_map_pts:
                idx = np.random.choice(len(acc_map), max_map_pts, replace=False)
                acc_map = acc_map[idx]
            map_history.append(acc_map.copy())
            keep_frame_iv_idx.append(len(iv_poses) - 1)
        if (i + 1) % 200 == 0:
            print(f"  [{seq_key}] frame {i+1}/{n}  map_pts={len(acc_map)}")

    # GenZ in one subprocess shot
    genz_poses = run_genz_on_clouds(genz_clouds, voxel_size=0.5, max_range=80.0,
                                     min_range=0.5, min_motion_th=0.5,
                                     initial_threshold=1.5, planarity_th=0.1)
    # Prepend identity so length = n+1 matches IV/KISS convention
    genz_poses = [np.eye(4)] + genz_poses

    # Build keyframe-prefixed trajectories for rendering
    for kidx in keep_frame_iv_idx:
        iv_traj.append(np.array([P[:3, 3] for P in iv_poses[:kidx + 1]]))
        kiss_traj.append(np.array([P[:3, 3] for P in kiss_poses[:kidx + 1]]))

    # Match GenZ by keyframe index (capped to its length)
    genz_traj = []
    for kidx in keep_frame_iv_idx:
        j = min(kidx + 1, len(genz_poses))
        genz_traj.append(np.array([P[:3, 3] for P in genz_poses[:j]]))

    print(f"[{seq_key}] done. {len(map_history)} keyframes (IV={len(iv_poses)-1}, "
          f"KISS={len(kiss_poses)-1}, GenZ={len(genz_poses)-1})")
    return map_history, iv_traj, kiss_traj, genz_traj, spec


def render_geode_3way_gif(seq_key, keyframes=120, fps=15, save_teaser=False):
    import imageio.v2 as imageio
    map_history, iv_traj, kiss_traj, genz_traj, spec = \
        build_geode_3way(seq_key, keyframes=keyframes)
    final_map = map_history[-1]
    full_iv = iv_traj[-1]
    full_kiss = kiss_traj[-1]
    full_genz = genz_traj[-1]

    all_x = np.concatenate([final_map[:, 0], full_iv[:, 0],
                            full_kiss[:, 0], full_genz[:, 0]])
    all_y = np.concatenate([final_map[:, 1], full_iv[:, 1],
                            full_kiss[:, 1], full_genz[:, 1]])
    pad = 15
    xmin, xmax = all_x.min() - pad, all_x.max() + pad
    ymin, ymax = all_y.min() - pad, all_y.max() + pad
    w, h = xmax - xmin, ymax - ymin
    target = 640 / 360
    if w / h > target:
        extra = w / target - h
        ymin -= extra / 2; ymax += extra / 2
    else:
        extra = h * target - w
        xmin -= extra / 2; xmax += extra / 2

    delta_kiss = 100 * (spec["iv_ate"] - spec["kiss_ate"]) / spec["kiss_ate"]
    delta_genz = 100 * (spec["iv_ate"] - spec["genz_ate"]) / spec["genz_ate"]
    title_line2 = (f"IV-GICP: {spec['iv_ate']:.2f}m  |  "
                   f"KISS: {spec['kiss_ate']:.2f}m ({delta_kiss:+.0f}%)  |  "
                   f"GenZ: {spec['genz_ate']:.2f}m ({delta_genz:+.0f}%)")

    iv_label = f"IV-GICP (ours): {spec['iv_ate']:.2f}m"
    kiss_label = f"KISS-ICP: {spec['kiss_ate']:.2f}m"
    genz_label = f"GenZ-ICP: {spec['genz_ate']:.2f}m"

    frames = []
    for i, (pts, iv, ki, gz) in enumerate(
            zip(map_history, iv_traj, kiss_traj, genz_traj)):
        title = (f"GEODE {spec['name']}  |  kf {i+1}/{len(map_history)}\n"
                 f"{title_line2}")
        img = render_frame(pts, iv[:, :2], ki[:, :2],
                           (xmin, xmax, ymin, ymax),
                           iv[-1, :2], color_mode="height",
                           ptsize=0.4, title=title,
                           traj_genz_xy=gz[:, :2],
                           iv_label=iv_label, kiss_label=kiss_label,
                           genz_label=genz_label)
        frames.append(img)
        if (i + 1) % 20 == 0:
            print(f"  [{seq_key}-render] {i+1}/{len(map_history)}")

    out = OUT_DIR / f"{seq_key}_3way_demo.gif"
    imageio.mimsave(str(out), frames, fps=fps, loop=0)
    print(f"\n[{seq_key}] saved → {out}  ({len(frames)} frames, "
          f"{out.stat().st_size / 1e6:.1f} MB)")

    if save_teaser:
        # Dump cache for gen_new_figures.py teaser
        npz_path = OUT_DIR / f"{seq_key}_teaser.npz"
        np.savez_compressed(str(npz_path),
                            map_pts=final_map,
                            iv_traj=full_iv,
                            kiss_traj=full_kiss,
                            genz_traj=full_genz,
                            iv_ate=spec["iv_ate"],
                            kiss_ate=spec["kiss_ate"],
                            genz_ate=spec["genz_ate"])
        print(f"[{seq_key}] teaser cache saved → {npz_path}")
    return out


# ── GEODE Urban_Tunnel02 (legacy 2-way, kept for backward compat) ─────────────

def build_geode02(keyframes=120, map_voxel=0.4, render_voxel=0.2,
                  max_map_pts=120_000):
    """Run IV-GICP + KISS-ICP on Urban_Tunnel02, accumulate map."""
    from run_geode_eval import (read_geode_frames, load_geode_gt,
                                 interpolate_gt_at_lidar_times,
                                 GEODE_ROOT, compose_poses)
    from iv_gicp.pipeline import IVGICPPipeline
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    bag_path = GEODE_ROOT / "sensor_data" / "Urban_tunnel" / "Urban_Tunnel02" / "Urban_Tunnel02.bag"
    print(f"[geode02] loading {bag_path}")
    frames = read_geode_frames(bag_path, max_frames=None)
    n = len(frames)
    print(f"[geode02] {n} frames loaded")

    pipeline = IVGICPPipeline(
        voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
        max_correspondence_distance=2.0, initial_threshold=1.5,
        min_motion_th=0.5, max_iterations=12, max_map_frames=500,
        map_radius=80.0, use_fim_weight=True, auto_alpha=False,
        auto_alpha_from_intensity=False, source_drop_small_voxels=False,
        source_max_output_features=0, source_min_feature_score=0.0,
        max_source_points=0, device="cpu",
    )

    cfg = KISSConfig()
    cfg.data.deskew = False; cfg.data.max_range = 80.0; cfg.data.min_range = 0.5
    cfg.mapping.voxel_size = 0.5
    kiss = KissICP(config=cfg)

    iv_poses = [np.eye(4)]
    kiss_poses = [np.eye(4)]
    rel_poses = []

    step = max(1, n // keyframes)
    keep_idx = set(range(0, n, step))
    acc_map = np.empty((0, 4))
    map_history = []
    iv_traj = []
    kiss_traj = []

    for i, (ts, pts) in enumerate(frames):
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        rel_poses.append(np.linalg.inv(iv_poses[-1]) @ result.pose)
        iv_poses.append(result.pose.copy())
        kiss.register_frame(pts[:, :3].astype(np.float64), np.full(len(pts), ts))
        kiss_poses.append(kiss.last_pose.copy())

        if i in keep_idx or i == n - 1:
            Tw = result.pose
            p = pts[:, :3].astype(np.float32)
            intensity = pts[:, 3].astype(np.float32)
            r = np.linalg.norm(p, axis=1)
            m = (r > 0.5) & (r < 60.0)
            p, intensity = p[m], intensity[m]
            pw = (Tw[:3, :3] @ p.T).T + Tw[:3, 3]
            combined = voxel_downsample(
                np.column_stack([pw, intensity]), render_voxel)
            acc_map = np.concatenate([acc_map, combined], axis=0)
            acc_map = voxel_downsample(acc_map, map_voxel)
            if len(acc_map) > max_map_pts:
                idx = np.random.choice(len(acc_map), max_map_pts, replace=False)
                acc_map = acc_map[idx]
            map_history.append(acc_map.copy())
            iv_traj.append(np.array([P[:3, 3] for P in iv_poses]))
            kiss_traj.append(np.array([P[:3, 3] for P in kiss_poses]))

        if (i + 1) % 200 == 0:
            print(f"  [geode02] frame {i+1}/{n}  map_pts={len(acc_map)}")

    print(f"[geode02] done. {len(map_history)} keyframes")
    return map_history, iv_traj, kiss_traj


def render_geode02_gif(keyframes=120, fps=15):
    import imageio.v2 as imageio
    map_history, iv_traj, kiss_traj = build_geode02(keyframes=keyframes)
    final = map_history[-1]
    full_iv = iv_traj[-1]
    full_kiss = kiss_traj[-1]
    # bounds from union of final map + both full trajectories
    all_x = np.concatenate([final[:, 0], full_iv[:, 0], full_kiss[:, 0]])
    all_y = np.concatenate([final[:, 1], full_iv[:, 1], full_kiss[:, 1]])
    pad = 15
    xmin, xmax = all_x.min() - pad, all_x.max() + pad
    ymin, ymax = all_y.min() - pad, all_y.max() + pad
    w, h = xmax - xmin, ymax - ymin
    target = 640 / 360
    if w / h > target:
        extra = w / target - h
        ymin -= extra / 2; ymax += extra / 2
    else:
        extra = h * target - w
        xmin -= extra / 2; xmax += extra / 2

    frames = []
    for i, (pts, iv, ki) in enumerate(zip(map_history, iv_traj, kiss_traj)):
        title = (f"GEODE Urban_Tunnel02 | kf {i+1}/{len(map_history)}\n"
                 f"IV-GICP (ours): 1.90m ATE  |  KISS-ICP: 13.70m (-86%)")
        img = render_frame(pts, iv[:, :2], ki[:, :2],
                           (xmin, xmax, ymin, ymax),
                           iv[-1, :2], color_mode="height",
                           ptsize=0.4, title=title)
        frames.append(img)
        if (i+1) % 20 == 0:
            print(f"  [geode02-render] {i+1}/{len(map_history)}")
    out = OUT_DIR / "geode_tunnel02_demo.gif"
    imageio.mimsave(str(out), frames, fps=fps, loop=0)
    print(f"\n[geode02] saved → {out}  ({len(frames)} frames, {out.stat().st_size/1e6:.1f} MB)")
    return out


# ── Generic 2-way / 3-way compositing helpers ────────────────────────────────

def _compute_view_bounds(final_map, trajs, pad=15):
    xs = [final_map[:, 0]] + [t[:, 0] for t in trajs]
    ys = [final_map[:, 1]] + [t[:, 1] for t in trajs]
    all_x = np.concatenate(xs); all_y = np.concatenate(ys)
    xmin, xmax = all_x.min() - pad, all_x.max() + pad
    ymin, ymax = all_y.min() - pad, all_y.max() + pad
    w, h = xmax - xmin, ymax - ymin
    target = 640 / 360
    if w / h > target:
        extra = w / target - h
        ymin -= extra / 2; ymax += extra / 2
    else:
        extra = h * target - w
        xmin -= extra / 2; xmax += extra / 2
    return xmin, xmax, ymin, ymax


def render_compare_gif(out_name, map_history, iv_traj, kiss_traj,
                       spec, genz_traj=None, fps=15, pad=15,
                       ptsize=0.4, color_mode="height"):
    """Generic renderer for 2-way / 3-way comparison GIFs."""
    import imageio.v2 as imageio
    trajs = [iv_traj[-1], kiss_traj[-1]]
    if genz_traj is not None:
        trajs.append(genz_traj[-1])
    bounds = _compute_view_bounds(map_history[-1], trajs, pad=pad)

    delta_kiss = 100 * (spec["iv_ate"] - spec["kiss_ate"]) / spec["kiss_ate"]
    iv_label = f"IV-GICP (ours): {spec['iv_ate']:.2f}m"
    kiss_label = f"KISS-ICP: {spec['kiss_ate']:.2f}m"
    genz_label = None
    if genz_traj is not None:
        genz_label = f"GenZ-ICP: {spec['genz_ate']:.2f}m"
        delta_genz = 100 * (spec["iv_ate"] - spec["genz_ate"]) / spec["genz_ate"]
        title_line2 = (f"IV-GICP: {spec['iv_ate']:.2f}m  |  "
                       f"KISS: {spec['kiss_ate']:.2f}m ({delta_kiss:+.0f}%)  |  "
                       f"GenZ: {spec['genz_ate']:.2f}m ({delta_genz:+.0f}%)")
    else:
        title_line2 = (f"IV-GICP (ours): {spec['iv_ate']:.2f}m  |  "
                       f"KISS-ICP: {spec['kiss_ate']:.2f}m ({delta_kiss:+.0f}%)")

    iters = (range(len(map_history)), map_history, iv_traj, kiss_traj)
    gz_iter = genz_traj if genz_traj is not None else [None] * len(map_history)
    frames = []
    for i, pts, iv, ki, gz in zip(*iters, gz_iter):
        title = f"{spec['name']}  |  kf {i+1}/{len(map_history)}\n{title_line2}"
        img = render_frame(pts, iv[:, :2], ki[:, :2],
                           bounds, iv[-1, :2], color_mode=color_mode,
                           ptsize=ptsize, title=title,
                           traj_genz_xy=gz[:, :2] if gz is not None else None,
                           iv_label=iv_label, kiss_label=kiss_label,
                           genz_label=genz_label)
        frames.append(img)
        if (i + 1) % 20 == 0:
            print(f"  [{out_name}-render] {i+1}/{len(map_history)}", flush=True)

    out = OUT_DIR / f"{out_name}_demo.gif"
    imageio.mimsave(str(out), frames, fps=fps, loop=0)
    print(f"\n[{out_name}] saved → {out}  ({len(frames)} frames, "
          f"{out.stat().st_size / 1e6:.1f} MB)")
    return out


def _accumulate_keyframe_map(iv_poses_list, keep_frame_iv_idx):
    """Build per-keyframe trajectory list from absolute poses + keyframe indices."""
    trajs = []
    for kidx in keep_frame_iv_idx:
        trajs.append(np.array([P[:3, 3] for P in iv_poses_list[:kidx + 1]]))
    return trajs


# ── HeLiPR 2-way (KAIST05, RIVER04) ──────────────────────────────────────────

HELIPR_SPECS = {
    "helipr_kaist05": dict(name="HeLiPR KAIST05 (Ouster)", seq="KAIST05",
                           iv_ate=0.289, kiss_ate=0.626),
    "helipr_river04": dict(name="HeLiPR RIVER04 (Ouster)", seq="RIVER04",
                           iv_ate=0.601, kiss_ate=0.899),
}


def build_helipr_2way(seq_key, keyframes=120, max_frames=2000,
                      map_voxel=0.5, render_voxel=0.25, max_map_pts=120_000):
    """Run IV-GICP + KISS-ICP on a HeLiPR Ouster sequence, accumulate IV map."""
    from run_helipr_eval import load_sequence
    from iv_gicp.pipeline import IVGICPPipeline
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    spec = HELIPR_SPECS[seq_key]
    frames_all, _, _, scan_ts = load_sequence(spec["seq"], max_frames=max_frames)
    n = len(frames_all)
    print(f"[{seq_key}] {n} frames loaded")

    pipeline = IVGICPPipeline(
        voxel_size=1.0, source_voxel_size=0.3, alpha=0.0,
        max_correspondence_distance=2.0, initial_threshold=2.0,
        min_motion_th=0.5, max_iterations=20, max_map_frames=20,
        map_radius=None, use_fim_weight=True, auto_alpha=False,
        auto_alpha_from_intensity=False, source_drop_small_voxels=False,
        source_max_output_features=0, source_min_feature_score=0.0,
        max_source_points=0, device="cpu",
    )
    cfg = KISSConfig()
    cfg.data.deskew = False; cfg.data.max_range = 80.0; cfg.data.min_range = 0.5
    cfg.mapping.voxel_size = 1.0
    kiss = KissICP(config=cfg)

    iv_poses = [np.eye(4)]
    kiss_poses = [np.eye(4)]
    step = max(1, n // keyframes)
    keep_idx = set(range(0, n, step))
    acc_map = np.empty((0, 4))
    map_history = []
    keep_frame_iv_idx = []

    for i, pts in enumerate(frames_all):
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=float(scan_ts[i]))
        iv_poses.append(result.pose.copy())
        kiss.register_frame(pts[:, :3].astype(np.float64), np.full(len(pts), float(i)))
        kiss_poses.append(kiss.last_pose.copy())

        if i in keep_idx or i == n - 1:
            Tw = result.pose
            p = pts[:, :3].astype(np.float32)
            intensity = pts[:, 3].astype(np.float32)
            r = np.linalg.norm(p, axis=1)
            m = (r > 0.5) & (r < 60.0)
            p, intensity = p[m], intensity[m]
            pw = (Tw[:3, :3] @ p.T).T + Tw[:3, 3]
            combined = voxel_downsample(
                np.column_stack([pw, intensity]), render_voxel)
            acc_map = np.concatenate([acc_map, combined], axis=0)
            acc_map = voxel_downsample(acc_map, map_voxel)
            if len(acc_map) > max_map_pts:
                idx = np.random.choice(len(acc_map), max_map_pts, replace=False)
                acc_map = acc_map[idx]
            map_history.append(acc_map.copy())
            keep_frame_iv_idx.append(len(iv_poses) - 1)
        if (i + 1) % 200 == 0:
            print(f"  [{seq_key}] frame {i+1}/{n}  map_pts={len(acc_map)}",
                  flush=True)

    iv_traj = _accumulate_keyframe_map(iv_poses, keep_frame_iv_idx)
    kiss_traj = _accumulate_keyframe_map(kiss_poses, keep_frame_iv_idx)
    print(f"[{seq_key}] done. {len(map_history)} keyframes")
    return map_history, iv_traj, kiss_traj, spec


def render_helipr_2way_gif(seq_key, keyframes=120, max_frames=2000, fps=15):
    map_history, iv_traj, kiss_traj, spec = build_helipr_2way(
        seq_key, keyframes=keyframes, max_frames=max_frames)
    return render_compare_gif(seq_key, map_history, iv_traj, kiss_traj,
                               spec, fps=fps)


# ── SubT Urban_UGV1 3-way ────────────────────────────────────────────────────

SUBT_URBAN1_SPEC = dict(name="SubT-MRS Urban_UGV1 (VLP-16)",
                        iv_ate=0.274, kiss_ate=0.285, genz_ate=0.286)


def build_subt_urban1_3way(keyframes=120, max_frames=None, map_voxel=0.3,
                           render_voxel=0.15, max_map_pts=120_000):
    from run_subt_eval import DATASETS, BASE_DIR, load_frames_from_zipped_bags
    from iv_gicp.pipeline import IVGICPPipeline
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    import os

    info = DATASETS["Urban_UGV1"]
    rosbag_zip = os.path.join(BASE_DIR, info["rosbag"])
    print(f"[subt_urban1] loading {rosbag_zip}")
    frames_all = []
    ts_all = []
    for ts, cloud in load_frames_from_zipped_bags(rosbag_zip, max_frames=max_frames):
        frames_all.append(cloud)
        ts_all.append(ts)
    n = len(frames_all)
    print(f"[subt_urban1] {n} frames loaded")

    pipeline = IVGICPPipeline(
        voxel_size=0.5, source_voxel_size=0.3, alpha=0.1,
        max_correspondence_distance=2.0, initial_threshold=2.0,
        min_motion_th=0.5, max_iterations=12, max_map_frames=200,
        map_radius=200.0, use_fim_weight=False, auto_alpha=False,
        auto_alpha_from_intensity=False, source_drop_small_voxels=False,
        source_max_output_features=0, source_min_feature_score=0.0,
        max_source_points=0, device="cpu",
    )
    cfg = KISSConfig()
    cfg.data.deskew = False; cfg.data.max_range = 80.0; cfg.data.min_range = 0.5
    cfg.mapping.voxel_size = 0.5
    kiss = KissICP(config=cfg)

    iv_poses = [np.eye(4)]
    kiss_poses = [np.eye(4)]
    genz_clouds = []
    step = max(1, n // keyframes)
    keep_idx = set(range(0, n, step))
    acc_map = np.empty((0, 4))
    map_history = []
    keep_frame_iv_idx = []

    for i, cloud in enumerate(frames_all):
        pts = cloud[:, :3].astype(np.float64)
        intensity = (cloud[:, 3] if cloud.shape[1] >= 4
                     else np.zeros(len(cloud), dtype=np.float64)).astype(np.float64)
        result = pipeline.process_frame(pts, intensity, timestamp=float(i))
        iv_poses.append(result.pose.copy())
        kiss.register_frame(pts, np.full(len(pts), float(i)))
        kiss_poses.append(kiss.last_pose.copy())
        genz_clouds.append(np.column_stack([pts, intensity]).astype(np.float32))

        if i in keep_idx or i == n - 1:
            Tw = result.pose
            p = pts.astype(np.float32)
            r = np.linalg.norm(p, axis=1)
            m = (r > 0.5) & (r < 40.0)
            p, inten = p[m], intensity[m].astype(np.float32)
            pw = (Tw[:3, :3] @ p.T).T + Tw[:3, 3]
            combined = voxel_downsample(
                np.column_stack([pw, inten]), render_voxel)
            acc_map = np.concatenate([acc_map, combined], axis=0)
            acc_map = voxel_downsample(acc_map, map_voxel)
            if len(acc_map) > max_map_pts:
                idx = np.random.choice(len(acc_map), max_map_pts, replace=False)
                acc_map = acc_map[idx]
            map_history.append(acc_map.copy())
            keep_frame_iv_idx.append(len(iv_poses) - 1)
        if (i + 1) % 200 == 0:
            print(f"  [subt_urban1] frame {i+1}/{n}  map_pts={len(acc_map)}",
                  flush=True)

    genz_poses = run_genz_on_clouds(genz_clouds, voxel_size=0.5, max_range=80.0,
                                     min_range=0.5, min_motion_th=0.5,
                                     initial_threshold=1.5, planarity_th=0.1)
    genz_poses = [np.eye(4)] + genz_poses

    iv_traj = _accumulate_keyframe_map(iv_poses, keep_frame_iv_idx)
    kiss_traj = _accumulate_keyframe_map(kiss_poses, keep_frame_iv_idx)
    genz_traj = []
    for kidx in keep_frame_iv_idx:
        j = min(kidx + 1, len(genz_poses))
        genz_traj.append(np.array([P[:3, 3] for P in genz_poses[:j]]))

    print(f"[subt_urban1] done. {len(map_history)} keyframes")
    return map_history, iv_traj, kiss_traj, genz_traj, SUBT_URBAN1_SPEC


def render_subt_urban1_3way_gif(keyframes=120, max_frames=None, fps=15):
    mh, iv, ki, gz, spec = build_subt_urban1_3way(
        keyframes=keyframes, max_frames=max_frames)
    return render_compare_gif("subt_urban1_3way", mh, iv, ki, spec,
                               genz_traj=gz, fps=fps, ptsize=0.5)


# ── KITTI seq02 full-seq 2-way ───────────────────────────────────────────────

KITTI_SPECS = {
    "kitti02": dict(name="KITTI seq02 (full-seq)", seq="02",
                    iv_ate=0.615, kiss_ate=0.807),
}


def build_kitti_2way(seq_key, keyframes=120, max_frames=None,
                     map_voxel=0.5, render_voxel=0.3, max_map_pts=120_000):
    from run_kitti_benchmark import load_kitti_sequence
    from iv_gicp.pipeline import IVGICPPipeline
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    spec = KITTI_SPECS[seq_key]
    frames_all, _ = load_kitti_sequence(spec["seq"], max_frames=max_frames)
    n = len(frames_all)
    print(f"[{seq_key}] {n} frames loaded")

    pipeline = IVGICPPipeline(
        voxel_size=1.0, source_voxel_size=0.3, alpha=0.1,
        max_correspondence_distance=2.0, initial_threshold=2.0,
        min_motion_th=0.5, max_iterations=12, max_map_frames=500,
        map_radius=None, use_fim_weight=False, auto_alpha=False,
        auto_alpha_from_intensity=False, source_drop_small_voxels=False,
        source_max_output_features=0, source_min_feature_score=0.0,
        max_source_points=0, device="cpu",
    )
    cfg = KISSConfig()
    cfg.data.deskew = False; cfg.data.max_range = 80.0; cfg.data.min_range = 0.5
    cfg.mapping.voxel_size = 1.0
    kiss = KissICP(config=cfg)

    iv_poses = [np.eye(4)]
    kiss_poses = [np.eye(4)]
    step = max(1, n // keyframes)
    keep_idx = set(range(0, n, step))
    acc_map = np.empty((0, 4))
    map_history = []
    keep_frame_iv_idx = []

    for i, f in enumerate(frames_all):
        result = pipeline.process_frame(f[:, :3], f[:, 3], timestamp=float(i))
        iv_poses.append(result.pose.copy())
        kiss.register_frame(f[:, :3].astype(np.float64), np.full(len(f), float(i)))
        kiss_poses.append(kiss.last_pose.copy())

        if i in keep_idx or i == n - 1:
            Tw = result.pose
            p = f[:, :3].astype(np.float32)
            intensity = f[:, 3].astype(np.float32)
            r = np.linalg.norm(p, axis=1)
            m = (r > 0.5) & (r < 60.0)
            p, intensity = p[m], intensity[m]
            pw = (Tw[:3, :3] @ p.T).T + Tw[:3, 3]
            combined = voxel_downsample(
                np.column_stack([pw, intensity]), render_voxel)
            acc_map = np.concatenate([acc_map, combined], axis=0)
            acc_map = voxel_downsample(acc_map, map_voxel)
            if len(acc_map) > max_map_pts:
                idx = np.random.choice(len(acc_map), max_map_pts, replace=False)
                acc_map = acc_map[idx]
            map_history.append(acc_map.copy())
            keep_frame_iv_idx.append(len(iv_poses) - 1)
        if (i + 1) % 200 == 0:
            print(f"  [{seq_key}] frame {i+1}/{n}  map_pts={len(acc_map)}",
                  flush=True)

    iv_traj = _accumulate_keyframe_map(iv_poses, keep_frame_iv_idx)
    kiss_traj = _accumulate_keyframe_map(kiss_poses, keep_frame_iv_idx)
    print(f"[{seq_key}] done. {len(map_history)} keyframes")
    return map_history, iv_traj, kiss_traj, spec


def render_kitti_2way_gif(seq_key, keyframes=120, max_frames=None, fps=15):
    mh, iv, ki, spec = build_kitti_2way(
        seq_key, keyframes=keyframes, max_frames=max_frames)
    return render_compare_gif(seq_key, mh, iv, ki, spec, fps=fps)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True,
                    choices=["subt", "geode02", "both",
                             "geode01_3way", "geode02_3way", "geode03_3way",
                             "geode_all_3way",
                             "helipr_kaist05", "helipr_river04",
                             "subt_urban1_3way", "kitti02"])
    ap.add_argument("--keyframes", type=int, default=120)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap number of frames to load (only for helipr/subt/kitti builders)")
    ap.add_argument("--save-teaser", action="store_true",
                    help="On geode02_3way, also dump NPZ cache for gen_new_figures.py teaser")
    args = ap.parse_args()

    if args.seq in ("subt", "both"):
        render_subt_gif(keyframes=args.keyframes, fps=args.fps)
    if args.seq in ("geode02", "both"):
        render_geode02_gif(keyframes=args.keyframes, fps=args.fps)
    if args.seq == "geode01_3way":
        render_geode_3way_gif("geode01", keyframes=args.keyframes, fps=args.fps)
    if args.seq == "geode02_3way":
        render_geode_3way_gif("geode02", keyframes=args.keyframes, fps=args.fps,
                              save_teaser=args.save_teaser)
    if args.seq == "geode03_3way":
        render_geode_3way_gif("geode03", keyframes=args.keyframes, fps=args.fps)
    if args.seq == "geode_all_3way":
        for s in ("geode01", "geode02", "geode03"):
            render_geode_3way_gif(s, keyframes=args.keyframes, fps=args.fps,
                                  save_teaser=(s == "geode02" and args.save_teaser))
    if args.seq in ("helipr_kaist05", "helipr_river04"):
        render_helipr_2way_gif(args.seq, keyframes=args.keyframes,
                                max_frames=args.max_frames or 2000,
                                fps=args.fps)
    if args.seq == "subt_urban1_3way":
        render_subt_urban1_3way_gif(keyframes=args.keyframes,
                                     max_frames=args.max_frames, fps=args.fps)
    if args.seq == "kitti02":
        render_kitti_2way_gif("kitti02", keyframes=args.keyframes,
                               max_frames=args.max_frames, fps=args.fps)


if __name__ == "__main__":
    main()

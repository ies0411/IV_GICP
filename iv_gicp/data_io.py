"""
데이터 입출력: synthetic point cloud 저장/로드
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


def save_synthetic_sequence(
    frames: List[np.ndarray],
    poses_gt: List[np.ndarray],
    output_dir: str,
) -> None:
    """
    합성 시퀀스 저장.
    - output_dir/pointclouds/frame_00000.npy, ... (x,y,z,intensity)
    - output_dir/poses_gt.npy (N,4,4)
    - output_dir/poses_gt.txt (KITTI format: tx ty tz ...)
    """
    out = Path(output_dir)
    pc_dir = out / "pointclouds"
    pc_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        np.save(pc_dir / f"frame_{i:05d}.npy", frame)
    poses_arr = np.array(poses_gt)
    np.save(out / "poses_gt.npy", poses_arr)
    with open(out / "poses_gt.txt", "w") as f:
        for T in poses_gt:
            t = T[:3, 3]
            f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")
    print(f"Saved: {len(frames)} point clouds, {len(poses_gt)} poses → {out}/")


def save_synthetic_sequence_pcd(
    frames: List[np.ndarray],
    poses_gt: List[np.ndarray],
    output_dir: str,
) -> None:
    """PCD 형식으로도 저장 (Open3D 사용)."""
    try:
        import open3d as o3d
    except ImportError:
        return
    out = Path(output_dir)
    pc_dir = out / "pointclouds_pcd"
    pc_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame[:, :3])
        if frame.shape[1] >= 4:
            c = np.tile(frame[:, 3:4] / 255.0, (1, 3))
            pcd.colors = o3d.utility.Vector3dVector(np.clip(c, 0, 1))
        o3d.io.write_point_cloud(str(pc_dir / f"frame_{i:05d}.pcd"), pcd)
    print(f"Saved PCD: {pc_dir}/")


def load_synthetic_sequence(data_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """저장된 시퀀스 로드. Returns (frames, poses_gt)."""
    data = Path(data_dir)
    frames = []
    i = 0
    while (data / "pointclouds" / f"frame_{i:05d}.npy").exists():
        frames.append(np.load(data / "pointclouds" / f"frame_{i:05d}.npy"))
        i += 1
    poses_gt = list(np.load(data / "poses_gt.npy"))
    return frames, poses_gt

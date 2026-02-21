"""
KITTI Data 로더: raw format (velodyne bin + oxts) 및 odometry benchmark format 지원

Supported formats:
  1. Raw:      data_root/velodyne_points/data/*.bin + oxts/data/*.txt
  2. Odometry: data_root/sequences/XX/velodyne/*.bin + poses/XX.txt
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


def load_velodyne_bin(path: str) -> np.ndarray:
    """
    KITTI velodyne .bin 로드.
    Format: N x 4 floats (x, y, z, intensity/re reflectance), little-endian.
    Returns: (N, 4) [x, y, z, I]
    """
    data = np.fromfile(path, dtype=np.float32)
    data = data.reshape(-1, 4)
    return data


def load_oxts_txt(path: str) -> np.ndarray:
    """oxts 한 프레임: lat, lon, alt, roll, pitch, yaw, ..."""
    with open(path) as f:
        line = f.readline()
    vals = [float(x) for x in line.split()]
    return np.array(vals)


def lat_to_scale(lat: float) -> float:
    return np.cos(lat * np.pi / 180.0)


def latlon_to_mercator(lat: float, lon: float, scale: float) -> Tuple[float, float]:
    er = 6378137.0
    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))
    return mx, my


def oxts_to_pose(oxts: np.ndarray, origin_oxts: Tuple[float, float] = (48.9843445, 8.4295857)) -> np.ndarray:
    """
    단일 oxts → 4x4 pose (world from sensor).
    oxts: [lat, lon, alt, roll, pitch, yaw, ...]
    """
    scale = lat_to_scale(origin_oxts[0])
    ox, oy = latlon_to_mercator(origin_oxts[0], origin_oxts[1], scale)
    origin = np.array([ox, oy, 0])

    tx, ty = latlon_to_mercator(oxts[0], oxts[1], scale)
    t = np.array([tx, ty, oxts[2]])
    t = t - origin

    rx, ry, rz = oxts[3], oxts[4], oxts[5]
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_kitti_odometry_sequence(
    data_root: str,
    sequence: str = "00",
    max_frames: Optional[int] = None,
    downsample: int = 1,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """
    Load KITTI odometry benchmark format (standard for leaderboard comparison).

    Expected directory structure:
        data_root/
            sequences/XX/velodyne/XXXXXX.bin
            poses/XX.txt  (12 floats per line = flattened 3×4 matrix)

    Args:
        data_root: path containing 'sequences/' and 'poses/' dirs
        sequence:  sequence ID (e.g. "00", "05", "08")
        max_frames: limit number of frames loaded
        downsample: stride downsample (1=all, 5=1/5)

    Returns:
        (frames, poses_gt) where frames is list of (N,4) [x,y,z,I]
        and poses_gt is list of 4×4 matrices (None if pose file missing)
    """
    root = Path(data_root)
    seq_dir = root / "sequences" / sequence / "velodyne"
    pose_file = root / "poses" / f"{sequence}.txt"

    if not seq_dir.exists():
        raise FileNotFoundError(f"Velodyne not found: {seq_dir}")

    bin_files = sorted(seq_dir.glob("*.bin"))
    if max_frames:
        bin_files = bin_files[:max_frames]

    frames = []
    for bf in bin_files:
        pc = load_velodyne_bin(str(bf))
        if downsample > 1:
            pc = pc[::downsample]
        intensity = pc[:, 3] if pc.shape[1] >= 4 else np.zeros(len(pc))
        frames.append(np.column_stack([pc[:, :3], intensity]))

    poses_gt = None
    if pose_file.exists():
        poses_gt = []
        with open(pose_file) as f:
            for line in f:
                vals = [float(x) for x in line.strip().split()]
                if len(vals) != 12:
                    continue
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses_gt.append(T)
        if max_frames and len(poses_gt) > max_frames:
            poses_gt = poses_gt[:max_frames]

    return frames, poses_gt


def load_kitti_sequence(
    data_root: str,
    max_frames: Optional[int] = None,
    downsample: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    KITTI raw sample 로드.
    data_root: data/kitti/sample
    Returns: (frames, poses_gt)
    """
    root = Path(data_root)
    velo_dir = root / "velodyne_points" / "data"
    oxts_dir = root / "oxts" / "data"

    if not velo_dir.exists():
        raise FileNotFoundError(f"Velodyne not found: {velo_dir}")

    bin_files = sorted(velo_dir.glob("*.bin"))
    if max_frames:
        bin_files = bin_files[:max_frames]

    frames = []
    poses_gt = []
    origin_oxts = None

    for i, bf in enumerate(bin_files):
        frame_id = bf.stem
        pc = load_velodyne_bin(str(bf))
        if downsample > 1:
            pc = pc[::downsample]
        if pc.shape[1] >= 4:
            intensity = pc[:, 3]
        else:
            intensity = np.zeros(len(pc))
        frames.append(np.column_stack([pc[:, :3], intensity]))

        oxts_file = oxts_dir / f"{frame_id}.txt"
        if oxts_file.exists():
            oxts = load_oxts_txt(str(oxts_file))
            if origin_oxts is None:
                origin_oxts = (oxts[0], oxts[1])
            T = oxts_to_pose(oxts, origin_oxts)
            poses_gt.append(T)
        else:
            poses_gt.append(np.eye(4))

    # GT 유효성: oxts에서 로드했고, 궤적에 이동이 있으면 유효
    traj_len = sum(np.linalg.norm(poses_gt[i][:3, 3] - poses_gt[i - 1][:3, 3]) for i in range(1, len(poses_gt)))
    if traj_len < 0.1:
        poses_gt = None  # GT 없음 또는 정지
    return frames, poses_gt

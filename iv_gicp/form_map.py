"""
FORM-style Map: Point-wise map update for comparison with Distribution Propagation.

FORM (Fixed-Lag Odometry with Reparative Mapping):
- Factor graph 최적화 후 pose가 T_old -> T_new로 변경
- 기존: 각 점 p를 p_new = T_new @ T_old^{-1} @ p 로 변환 (O(n) per frame)
- 문제: 점 10만 개면 10만 번 행렬 곱셈/벡터 변환
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .se3_utils import se3_inverse, se3_compose, transform_point


@dataclass
class FormPoint:
    """점 + frame_id (어느 pose로 변환된 점인지)."""

    xyz: np.ndarray  # (3,)
    intensity: float
    frame_id: int


class FormMap:
    """
    FORM-style point cloud map with frame_id tracking.
    Pose 변경 시 점 단위 변환 수행.
    """

    def __init__(self):
        self.points: np.ndarray = np.empty((0, 4))  # [x,y,z,I]
        self.frame_ids: np.ndarray = np.empty(0, dtype=np.int32)
        self.poses: List[np.ndarray] = []  # current poses (world from sensor)

    def add_points(
        self,
        points: np.ndarray,
        frame_id: int,
        pose: np.ndarray,
    ) -> None:
        """
        Add points in world frame (이미 pose로 변환된 상태).
        points: (N, 3) or (N, 4) [x,y,z,I]
        """
        pts = np.asarray(points)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        n = len(pts)
        if pts.shape[1] < 4:
            pts = np.column_stack([pts[:, :3], np.zeros(n)])
        self.points = np.vstack([self.points, pts]) if self.points.size else pts
        self.frame_ids = (
            np.concatenate([self.frame_ids, np.full(n, frame_id, dtype=np.int32)])
            if self.frame_ids.size
            else np.full(n, frame_id, dtype=np.int32)
        )
        while len(self.poses) <= frame_id:
            self.poses.append(np.eye(4))
        self.poses[frame_id] = np.array(pose, dtype=float)

    def update_poses_form_style(
        self,
        optimized_poses: List[np.ndarray],
    ) -> float:
        """
        FORM-style 점 단위 갱신.
        p_new = T_new @ T_old^{-1} @ p_old

        Returns: 수행한 변환 횟수 (벤치마크용)
        """
        n_transforms = 0
        for fid in range(min(len(self.poses), len(optimized_poses))):
            T_old = self.poses[fid]
            T_new = optimized_poses[fid]
            if np.allclose(T_old, T_new):
                continue
            mask = self.frame_ids == fid
            if not np.any(mask):
                continue
            delta_T = se3_compose(T_new, se3_inverse(T_old))
            pts = self.points[mask, :3]
            self.points[mask, :3] = transform_point(delta_T, pts)
            n_transforms += np.sum(mask)
            self.poses[fid] = T_new.copy()
        return float(n_transforms)

    def get_points(self) -> np.ndarray:
        return self.points.copy()

    def get_point_count(self) -> int:
        return len(self.points)

    def get_voxel_means_from_points(self, voxel_size: float = 0.5) -> List[Tuple[np.ndarray, int]]:
        """점들을 복셀화하여 (mean, frame_id) 목록 반환. (Distribution과 비교용)"""
        if len(self.points) == 0:
            return []
        xyz = self.points[:, :3]
        frame_ids = self.frame_ids
        keys = {}
        for i in range(len(xyz)):
            k = tuple((xyz[i] / voxel_size).astype(int))
            if k not in keys:
                keys[k] = []
            keys[k].append(i)
        out = []
        for k, indices in keys.items():
            pts = xyz[indices]
            fids = frame_ids[indices]
            mean = np.mean(pts, axis=0)
            fid = int(np.median(fids))  # 대표 frame_id
            out.append((mean, fid))
        return out

"""
Map Refinement: Factor Graph + FORM vs Distribution Propagation

Factor Graph 최적화 후 지도 갱신 방식 비교:
1. FORM: 점 단위 변환 p_new = T_new @ T_old^{-1} @ p  (O(n))
2. Distribution Propagation: 복셀 (μ, Σ)만 Lie theory로 업데이트 (O(k), k << n)
"""

import time
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from .factor_graph import PoseGraphOptimizer, FormMapUpdater
from .form_map import FormMap
from .distribution_propagation import DistributionPropagator, propagate_mean, propagate_covariance
from .se3_utils import se3_inverse, se3_compose


@dataclass
class RefinementResult:
    """지도 갱신 결과 (벤치마크용)."""

    method: str  # "form" or "distribution"
    n_updates: int  # 점 or 복셀 수
    time_ms: float
    poses_corrected: int


class MapRefinementBenchmark:
    """
    FORM vs Distribution Propagation 비교 벤치마크.
    """

    def __init__(
        self,
        lag_size: int = 10,
        n_points_per_frame: int = 10000,
        voxel_size: float = 0.5,
    ):
        self.lag_size = lag_size
        self.n_points_per_frame = n_points_per_frame
        self.voxel_size = voxel_size

    def run_form_update(
        self,
        form_map: FormMap,
        optimized_poses: List[np.ndarray],
    ) -> RefinementResult:
        """FORM 스타일 점 단위 갱신."""
        t0 = time.perf_counter()
        n_transforms = form_map.update_poses_form_style(optimized_poses)
        t1 = time.perf_counter()
        return RefinementResult(
            method="form",
            n_updates=int(n_transforms),
            time_ms=(t1 - t0) * 1000,
            poses_corrected=len(optimized_poses),
        )

    def run_distribution_propagation(
        self,
        voxel_means: List[np.ndarray],
        voxel_covs: List[np.ndarray],
        voxel_frame_ids: List[int],
        poses_old: List[np.ndarray],
        poses_new: List[np.ndarray],
    ) -> RefinementResult:
        """분포 전파: 복셀 (μ, Σ)만 업데이트."""
        propagator = DistributionPropagator()
        voxel_ids = list(range(len(voxel_means)))
        propagator.set_voxel_map(voxel_ids, voxel_means, voxel_covs, voxel_frame_ids)
        delta_per_frame = {}
        for fid in range(min(len(poses_old), len(poses_new))):
            T_old, T_new = poses_old[fid], poses_new[fid]
            if np.allclose(T_old, T_new):
                continue
            delta_per_frame[fid] = se3_compose(T_new, se3_inverse(T_old))
        t0 = time.perf_counter()
        propagator.propagate_per_frame(delta_per_frame)
        t1 = time.perf_counter()
        return RefinementResult(
            method="distribution",
            n_updates=len(voxel_means),
            time_ms=(t1 - t0) * 1000,
            poses_corrected=len(delta_per_frame),
        )

    def benchmark(
        self,
        n_frames: int = 20,
        randomize_correction: bool = True,
    ) -> Tuple[RefinementResult, RefinementResult]:
        """
        FORM vs Distribution Propagation 속도 비교.
        """
        np.random.seed(42)
        form_map = FormMap()
        voxel_means, voxel_covs, voxel_frame_ids = [], [], []

        # Build map with synthetic data
        poses = [np.eye(4)]
        for i in range(1, n_frames):
            T = np.eye(4)
            T[:3, 3] = np.random.randn(3) * 0.1 + np.array([i * 0.5, 0, 0])
            poses.append(T)
        poses_old = [p.copy() for p in poses]

        # Add points (in world frame)
        for fid in range(n_frames):
            pts_local = np.random.randn(self.n_points_per_frame, 3) * 2
            pts_local[:, 0] += 5
            pts_world = transform_points(poses[fid], pts_local)
            intensity = np.random.rand(self.n_points_per_frame) * 100
            form_map.add_points(
                np.column_stack([pts_world, intensity]),
                fid,
                poses[fid],
            )
            # Voxelize for distribution propagation (per frame)
            vox_idx = voxelize_indices(pts_world, self.voxel_size)
            for k, idx in vox_idx.items():
                pts_k = pts_world[idx]
                voxel_means.append(np.mean(pts_k, axis=0))
                cov = np.cov(pts_k.T) if len(pts_k) > 1 else np.eye(3) * 1e-6
                voxel_covs.append(cov + 1e-6 * np.eye(3))
                voxel_frame_ids.append(fid)

        # Simulate pose correction (small random delta)
        poses_new = [p.copy() for p in poses]
        if randomize_correction:
            for i in range(n_frames):
                delta = np.eye(4)
                delta[:3, 3] = np.random.randn(3) * 0.05
                delta[:3, :3] = np.eye(3) + 0.01 * np.random.randn(3, 3)
                delta[:3, :3] = delta[:3, :3] / np.linalg.norm(delta[:3, :3])
                poses_new[i] = se3_compose(poses_new[i], delta)

        res_form = self.run_form_update(form_map, poses_new)
        res_dist = self.run_distribution_propagation(
            voxel_means,
            voxel_covs,
            voxel_frame_ids,
            poses_old,
            poses_new,
        )
        return res_form, res_dist


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform (N,3) points by 4x4 T."""
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def voxelize_indices(xyz: np.ndarray, voxel_size: float) -> Dict[Tuple, np.ndarray]:
    """Returns dict of voxel_key -> indices."""
    keys = {}
    for i in range(len(xyz)):
        k = tuple((xyz[i] / voxel_size).astype(int))
        if k not in keys:
            keys[k] = []
        keys[k].append(i)
    return {k: np.array(v) for k, v in keys.items()}

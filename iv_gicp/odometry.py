"""
IV-GICP Public API — KISS-ICP style interface.

Minimal, preset-driven entry point for using IV-GICP as a library.

    from iv_gicp import IVGICPOdometry

    od = IVGICPOdometry()                  # default (outdoor)
    od = IVGICPOdometry.outdoor()          # KITTI / large-scale outdoor
    od = IVGICPOdometry.indoor()           # indoor rooms, corridors
    od = IVGICPOdometry.tunnel()           # degenerate: tunnels, mines
    od = IVGICPOdometry.metro()            # subway / long-corridor with loop

    for points in scan_sequence:
        pose = od.register_frame(points)   # (N,4) xyzi  or  (N,3) xyz

    od.poses       # list of (4,4) numpy arrays
    od.last_pose   # most recent (4,4) numpy array
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from .pipeline import IVGICPPipeline, OdometryResult


class IVGICPOdometry:
    """
    LiDAR odometry powered by IV-GICP.

    Args:
        voxel_size: Map voxel size in metres.  Larger = faster, less detail.
        alpha: Intensity weight (0 = pure geometry; >0 = geo+photometric).
        source_voxel_size: Input scan downsampling voxel size (m).  0 = no downsampling.
        map_radius: Spatial eviction radius (m).  None = age-based eviction.
        max_map_frames: How many recent frames to keep in the map.  None = auto.
        window_size: FORM-style window smoothing frames.  1 = disabled.
        adaptive_voxelization: C1 entropy-based voxel splitting (slower; helps tunnels).
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
    """

    def __init__(
        self,
        voxel_size: float = 1.0,
        alpha: float = 0.1,
        source_voxel_size: float = 0.3,
        max_correspondence_distance: float = 2.0,
        min_range: float = 0.5,
        max_range: float = 80.0,
        map_radius: Optional[float] = None,
        max_map_frames: Optional[int] = None,
        window_size: int = 1,
        adaptive_voxelization: bool = False,
        device: str = "auto",
    ) -> None:
        self._pipeline = IVGICPPipeline(
            voxel_size=voxel_size,
            alpha=alpha,
            source_voxel_size=source_voxel_size,
            max_correspondence_distance=max_correspondence_distance,
            min_range=min_range,
            max_range=max_range,
            map_radius=map_radius,
            max_map_frames=max_map_frames,
            window_size=window_size,
            adaptive_voxelization=adaptive_voxelization,
            device=device,
        )

    # ── Preset constructors ────────────────────────────────────────────────

    @classmethod
    def outdoor(cls, device: str = "auto") -> "IVGICPOdometry":
        """Large-scale outdoor (KITTI, urban driving).

        Best for sequences with varied geometry and good intensity contrast.
        """
        return cls(
            voxel_size=1.0,
            alpha=0.1,
            source_voxel_size=0.3,
            max_correspondence_distance=2.0,
            max_range=80.0,
            map_radius=None,
            device=device,
        )

    @classmethod
    def indoor(cls, device: str = "auto") -> "IVGICPOdometry":
        """Indoor rooms and corridors (Hilti, SubT mines).

        Smaller voxels + higher intensity weight for structured indoor scenes.
        """
        return cls(
            voxel_size=0.3,
            alpha=0.5,
            source_voxel_size=0.2,
            max_correspondence_distance=1.0,
            max_range=40.0,
            map_radius=40.0,
            device=device,
        )

    @classmethod
    def tunnel(cls, device: str = "auto") -> "IVGICPOdometry":
        """Degenerate geometry: tunnels, mines, corridors (GEODE Urban Tunnel, SubT).

        Pure geometry registration; spatial map eviction prevents old-map contamination.
        """
        return cls(
            voxel_size=0.5,
            alpha=0.0,
            source_voxel_size=0.25,
            max_correspondence_distance=1.5,
            max_range=80.0,
            map_radius=80.0,
            device=device,
        )

    @classmethod
    def metro(cls, device: str = "auto") -> "IVGICPOdometry":
        """Long-corridor subway / metro (GEODE Metro).

        Intensity-augmented with window smoothing to handle geometry degeneracy.
        """
        return cls(
            voxel_size=0.5,
            alpha=0.5,
            source_voxel_size=0.2,
            max_correspondence_distance=1.5,
            max_range=60.0,
            map_radius=60.0,
            window_size=10,
            device=device,
        )

    @classmethod
    def underground(cls, device: str = "auto") -> "IVGICPOdometry":
        """Underground mines and constrained environments (SubT, MulRan).

        Spatial eviction + moderate intensity weight.
        """
        return cls(
            voxel_size=0.5,
            alpha=0.1,
            source_voxel_size=0.3,
            max_correspondence_distance=2.0,
            max_range=50.0,
            map_radius=50.0,
            device=device,
        )

    # ── Core interface ─────────────────────────────────────────────────────

    def register_frame(
        self,
        points: np.ndarray,
        intensities: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> np.ndarray:
        """
        Register one LiDAR scan and return the absolute pose.

        Args:
            points: (N, 3) or (N, 4+) array.
                    If shape is (N, 4+), column 3 is used as intensity unless
                    ``intensities`` is provided explicitly.
            intensities: (N,) array of per-point intensity values (optional).
                         If omitted and ``points`` has a 4th column, that column
                         is used.  Otherwise defaults to zeros.
            timestamp: Scan timestamp in seconds (optional, stored in trajectory).

        Returns:
            pose: (4, 4) float64 SE(3) transform — world ← sensor.
        """
        result: OdometryResult = self._pipeline.process_frame(
            points=points,
            intensities=intensities,
            timestamp=timestamp,
        )
        return result.pose

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def poses(self) -> List[np.ndarray]:
        """All registered poses as a list of (4, 4) numpy arrays."""
        return self._pipeline.trajectory.poses

    @property
    def last_pose(self) -> np.ndarray:
        """Most recent (4, 4) pose.  Identity before first frame."""
        poses = self._pipeline.trajectory.poses
        return poses[-1] if poses else np.eye(4)

    @property
    def timestamps(self) -> List[float]:
        """Timestamps corresponding to each pose."""
        return self._pipeline.trajectory.timestamps

    @property
    def pipeline(self) -> IVGICPPipeline:
        """Direct access to the underlying IVGICPPipeline (advanced use)."""
        return self._pipeline

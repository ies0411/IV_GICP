"""
IV-GICP 시각화: 논문용 Figure 생성

- Trajectory (2D/3D)
- Point cloud map
- Adaptive voxel 시각화
- 파일 저장만 (display 불필요) → SSH/headless OK
"""

import matplotlib

matplotlib.use("Agg")  # headless: display 없이 파일 저장

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


def plot_trajectory_2d(
    poses: List[np.ndarray],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    poses_gt: Optional[List[np.ndarray]] = None,
) -> None:
    """
    2D 궤적 플롯 (top-down, 논문 Fig용).
    poses_gt: Ground truth 비교 시.
    """
    import matplotlib.pyplot as plt

    title = title or ("Trajectory: GT vs IV-GICP" if poses_gt else "Trajectory (Bird's Eye View)")
    xy = np.array([p[:2, 3] for p in poses])
    fig, ax = plt.subplots(figsize=(6, 6))
    if poses_gt is not None:
        xy_gt = np.array([p[:2, 3] for p in poses_gt])
        ax.plot(xy_gt[:, 0], xy_gt[:, 1], "g--", linewidth=2, label="Ground Truth")
    ax.plot(xy[:, 0], xy[:, 1], "b-", linewidth=2, label="IV-GICP")
    ax.scatter(xy[0, 0], xy[0, 1], c="g", s=100, marker="o", label="Start")
    ax.scatter(xy[-1, 0], xy[-1, 1], c="r", s=100, marker="s", label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_trajectory_3d(
    poses: List[np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Trajectory (3D)",
) -> None:
    """3D 궤적 플롯."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xyz = np.array([p[:3, 3] for p in poses])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "b-", linewidth=2)
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], c="g", s=100)
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], c="r", s=100)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_point_cloud_map(
    points: np.ndarray,
    poses: Optional[List[np.ndarray]] = None,
    save_path: Optional[str] = None,
    intensity_colormap: bool = True,
) -> None:
    """
    포인트 클라우드 + 궤적. intensity로 색상 (논문용).
    """
    import matplotlib.pyplot as plt

    pts = points[:, :3]
    if intensity_colormap and points.shape[1] >= 4:
        c = points[:, 3]
    else:
        c = pts[:, 2]
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=c, s=1, cmap="viridis", alpha=0.6)
    if poses:
        xy = np.array([p[:2, 3] for p in poses])
        ax.plot(xy[:, 0], xy[:, 1], "r-", linewidth=2, label="Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.set_title("Point Cloud Map (Bird's Eye View)")
    plt.colorbar(sc, ax=ax, label="Intensity" if intensity_colormap else "Z")
    if poses:
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_adaptive_voxels(
    voxels: List[Tuple[np.ndarray, np.ndarray, float, int]],
    save_path: Optional[str] = None,
    max_display: int = 2000,
) -> None:
    """
    Adaptive voxel 시각화 (복셀 중심 + 크기).
    기하학적 엔트로피가 높은 곳은 작은 복셀.
    """
    import matplotlib.pyplot as plt

    voxels = voxels[:max_display]
    if not voxels:
        return
    means = np.array([v[0] for v in voxels])
    covs = [v[1] for v in voxels]
    sizes = np.array([np.sqrt(np.linalg.det(c + 1e-6 * np.eye(3))) for c in covs])
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(means[:, 0], means[:, 1], c=sizes, s=5, cmap="plasma", alpha=0.7)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.set_title("Adaptive Voxelization (color = voxel size)")
    plt.colorbar(sc, ax=ax, label="Voxel size (det(Σ)^0.5)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_point_cloud_3d(
    points: np.ndarray,
    poses: Optional[List[np.ndarray]] = None,
    save_path: Optional[str] = None,
    downsample: int = 5,
) -> None:
    """
    3D 포인트 클라우드 + 궤적 저장 (matplotlib, SSH/headless OK).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    pts = points[::downsample, :3]  # downsample for faster rendering
    if points.shape[1] >= 4:
        c = points[::downsample, 3]
    else:
        c = pts[:, 2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=0.5, cmap="viridis", alpha=0.5)
    if poses:
        xyz = np.array([p[:3, 3] for p in poses])
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "r-", linewidth=2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Point Cloud Map (3D)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def _is_headless() -> bool:
    """DISPLAY 없으면 headless (SSH 등). Open3D OffscreenRenderer 실패 가능."""
    import os

    return not os.environ.get("DISPLAY", "").strip()


def save_open3d_3d_view(
    points: np.ndarray,
    poses: Optional[List[np.ndarray]] = None,
    save_path: Optional[str] = None,
    width: int = 1280,
    height: int = 960,
    downsample: int = 3,
) -> bool:
    """
    Open3D OffscreenRenderer로 3D 뷰 저장 (인터랙티브 뷰어와 유사한 품질).
    SSH/headless(DISPLAY 없음)에서는 skip → matplotlib fallback.
    """
    if _is_headless():
        return False
    try:
        import open3d as o3d
        import open3d.visualization.rendering as rendering
    except ImportError:
        return False

    try:
        pts = points[::downsample, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if points.shape[1] >= 4:
            i = points[::downsample, 3]
            i_norm = (i - i.min()) / (i.max() - i.min() + 1e-6)
            colors = np.tile(i_norm[:, np.newaxis], (1, 3))
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        else:
            pcd.paint_uniform_color([0.5, 0.5, 0.5])

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2.0

        render = rendering.OffscreenRenderer(width, height)
        render.scene.add_geometry("pcd", pcd, mat)

        # 궤적: LineSet (시작-끝 연결)
        if poses and len(poses) >= 2:
            xyz = np.array([p[:3, 3] for p in poses])
            lines = [[i, i + 1] for i in range(len(xyz) - 1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(xyz)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(lines), 1)))
            mat_line = rendering.MaterialRecord()
            mat_line.shader = "defaultUnlit"
            mat_line.line_width = 3.0
            render.scene.add_geometry("trajectory", line_set, mat_line)

        # Camera: 바운딩 박스 기반 시점
        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        dist = max(extent) * 2.5
        eye = center + np.array([dist * 0.7, dist * 0.5, dist * 0.7])
        render.setup_camera(60.0, eye, center, [0, 0, 1])
        render.scene.scene.set_sun_light([0.577, 0.577, 0.577], [1.0, 1.0, 1.0], 75000)
        render.scene.scene.enable_sun_light(True)
        render.scene.show_axes(True)

        img = render.render_to_image()
        if save_path:
            o3d.io.write_image(save_path, img, 9)
            print(f"Saved (Open3D): {save_path}")
        return True
    except Exception as e:
        if save_path:
            print(f"Open3D offscreen failed ({e}), using matplotlib fallback")
        return False


def save_all_figures(
    pipeline,
    output_dir: str = "output",
    fmt: str = "png",
    use_open3d_3d: bool = True,
    poses_gt: Optional[List[np.ndarray]] = None,
) -> None:
    """
    논문용 Figure 일괄 저장 (파일만, display 불필요 → SSH/headless OK).
    poses_gt: Ground truth 궤적 (GT vs Estimated 비교용).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext = f".{fmt}"
    traj = pipeline.get_trajectory()
    plot_trajectory_2d(
        traj.poses,
        str(out / f"trajectory_2d{ext}"),
        poses_gt=poses_gt,
    )
    plot_trajectory_3d(traj.poses, str(out / f"trajectory_3d{ext}"))
    if pipeline.accumulated_points is not None:
        plot_point_cloud_map(
            pipeline.accumulated_points,
            traj.poses,
            str(out / f"map_bev{ext}"),
        )
        # 3D: Open3D 시도 → 실패 시 matplotlib
        map_3d_path = str(out / f"map_3d{ext}")
        if use_open3d_3d and fmt == "png":
            ok = save_open3d_3d_view(
                pipeline.accumulated_points,
                traj.poses,
                map_3d_path,
            )
            if not ok:
                plot_point_cloud_3d(
                    pipeline.accumulated_points,
                    traj.poses,
                    map_3d_path,
                )
        else:
            plot_point_cloud_3d(
                pipeline.accumulated_points,
                traj.poses,
                map_3d_path,
            )
    if pipeline.map_voxels is not None:
        plot_adaptive_voxels(
            pipeline.map_voxels,
            str(out / f"adaptive_voxels{ext}"),
        )
    print(f"Figures saved to {out}/")

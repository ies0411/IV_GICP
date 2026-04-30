#!/usr/bin/env python3
"""
Generate per-frame Hessian condition number (κ_H) figure.

Shows the clear separation between degenerate environments (κ_H >> 100)
and well-conditioned environments (κ_H < 30), justifying C1 auto-gate.

Runs IV-GICP pipeline on GEODE Urban Tunnel01 and KITTI seq00 (200fr each),
extracting per-frame κ_H from OdometryResult.kappa.

Outputs:
  docs/figures/kappa_distribution.pdf
  docs/figures/kappa_distribution.png

Usage:
  uv run python examples/gen_kappa_figure.py [--max-frames 200]
"""
import sys
import argparse
import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "GEODE Urban Tunnel01": "#c62828",
    "KITTI seq00":           "#1565c0",
}


def run_geode(max_frames=200):
    """Run IV-GICP on GEODE Urban Tunnel01, return per-frame kappa."""
    from examples.run_geode_eval import read_geode_frames
    from iv_gicp.pipeline import IVGICPPipeline

    bag_path = Path("/home/km/deepai_dev_data/GEODE/sensor_data/Urban_tunnel"
                     "/Urban_Tunnel01/Urban_Tunnel01.bag")
    if not bag_path.exists():
        print(f"  GEODE bag not found: {bag_path}")
        return None

    frames = read_geode_frames(bag_path, max_frames=max_frames)
    pipe = IVGICPPipeline(
        voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
        max_correspondence_distance=2.0, max_iterations=12,
        map_radius=80.0, min_motion_th=0.5,
        use_fim_weight=True, min_range=0.5, max_range=80.0,
    )

    kappas = []
    for i, (ts, pts4) in enumerate(frames):
        xyz, ints = pts4[:, :3], pts4[:, 3]
        result = pipe.process_frame(xyz, ints)
        kappas.append(result.kappa)
        if (i + 1) % 50 == 0:
            print(f"  GEODE: {i+1}/{len(frames)} frames, κ={result.kappa:.1f}")

    return np.array(kappas)


def run_kitti(max_frames=200):
    """Run IV-GICP on KITTI seq00, return per-frame kappa."""
    from iv_gicp.pipeline import IVGICPPipeline

    velo_dir = Path("/home/km/deepai_dev_data/kitti/dataset"
                     "/sequences/00/velodyne")
    if not velo_dir.exists():
        print(f"  KITTI velo dir not found: {velo_dir}")
        return None

    bin_files = sorted(glob.glob(str(velo_dir / "*.bin")))[:max_frames]
    pipe = IVGICPPipeline(
        voxel_size=1.0, source_voxel_size=0.3, alpha=0.1,
        max_correspondence_distance=2.0, max_iterations=12,
        map_radius=None, min_motion_th=0.1,
        use_fim_weight=False, min_range=0.5, max_range=80.0,
    )

    kappas = []
    for i, bf in enumerate(bin_files):
        raw = np.fromfile(bf, dtype=np.float32).reshape(-1, 4)
        xyz, ints = raw[:, :3].astype(np.float64), raw[:, 3].astype(np.float64)
        result = pipe.process_frame(xyz, ints)
        kappas.append(result.kappa)
        if (i + 1) % 50 == 0:
            print(f"  KITTI: {i+1}/{len(bin_files)} frames, κ={result.kappa:.1f}")

    return np.array(kappas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()

    print(f"=== Condition number figure (max_frames={args.max_frames}) ===")
    results = {}

    print("\n--- GEODE Urban Tunnel01 ---")
    kappas = run_geode(args.max_frames)
    if kappas is not None:
        results["GEODE Urban Tunnel01"] = kappas
        print(f"  mean κ={kappas.mean():.1f}, median={np.median(kappas):.1f}, "
              f"max={kappas.max():.1f}, >100: {(kappas>100).sum()}/{len(kappas)}")

    print("\n--- KITTI seq00 ---")
    kappas = run_kitti(args.max_frames)
    if kappas is not None:
        results["KITTI seq00"] = kappas
        print(f"  mean κ={kappas.mean():.1f}, median={np.median(kappas):.1f}, "
              f"max={kappas.max():.1f}, >100: {(kappas>100).sum()}/{len(kappas)}")

    if not results:
        print("No data. Exiting.")
        return

    # === Figure ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # Left: per-frame κ_H (log scale)
    for label, kappas in results.items():
        color = COLORS[label]
        frames = np.arange(len(kappas))
        ax1.semilogy(frames, np.clip(kappas, 1, None), color=color, linewidth=0.8,
                      alpha=0.7, label=label)

    ax1.axhline(y=100, color="#888888", linestyle="--", linewidth=1.2, alpha=0.8,
                label="$\\kappa_H = 100$ (auto-gate)")
    ax1.fill_between([0, max(len(k) for k in results.values())],
                     100, 1e6, color="#ffcdd2", alpha=0.15)
    ax1.fill_between([0, max(len(k) for k in results.values())],
                     1, 100, color="#bbdefb", alpha=0.15)
    ax1.text(5, 300, "C1 ACTIVE\n(degenerate)", fontsize=7, color="#c62828", alpha=0.7)
    ax1.text(5, 10, "C1 INACTIVE\n(well-conditioned)", fontsize=7, color="#1565c0", alpha=0.7)
    ax1.set_xlabel("Frame", fontsize=9)
    ax1.set_ylabel("$\\kappa_H = \\lambda_{\\max} / \\lambda_{\\min}$", fontsize=9)
    ax1.set_title("Per-Frame Hessian Condition Number", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.tick_params(labelsize=8)
    ax1.set_ylim(1, None)

    # Right: histogram
    for label, kappas in results.items():
        color = COLORS[label]
        log_k = np.log10(np.clip(kappas, 1, None))
        ax2.hist(log_k, bins=40, color=color, alpha=0.5, density=True,
                 label=label, edgecolor=color, linewidth=0.5)

    ax2.axvline(x=2.0, color="#888888", linestyle="--", linewidth=1.2, alpha=0.8,
                label="$\\kappa_H = 100$")
    ax2.set_xlabel("$\\log_{10}(\\kappa_H)$", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("$\\kappa_H$ Distribution", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_path = OUT / f"kappa_distribution.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)

    # Save raw data for reference
    np.savez(OUT / "kappa_data.npz",
             **{label.replace(" ", "_"): kappas for label, kappas in results.items()})
    print("Saved: kappa_data.npz")


if __name__ == "__main__":
    main()

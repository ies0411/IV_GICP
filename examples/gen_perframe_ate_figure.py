#!/usr/bin/env python3
"""
Generate per-frame positional error evolution figure for IV-GICP paper.

Shows WHERE IV-GICP maintains accuracy while KISS/GenZ diverge on tunnel sequences.
This is the single most compelling visual evidence for the paper.

Outputs:
  docs/figures/perframe_ate_geode.pdf — per-frame error on GEODE Urban Tunnel 01 & 02
  docs/figures/perframe_ate_geode.png

Usage:
  uv run python examples/gen_perframe_ate_figure.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / "geode"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "GT":       "#333333",
    "IV-GICP":  "#1a73e8",
    "KISS-ICP": "#e8710a",
    "GenZ-ICP": "#34a853",
}


def load_tum(path):
    lines = [l for l in Path(path).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    data = np.array([list(map(float, l.split())) for l in lines])
    return data[:, 1:4]  # tx ty tz


def load_kitti_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            v = list(map(float, line.split()))
            if len(v) == 12:
                poses.append([v[3], v[7], v[11]])
            elif len(v) >= 16:
                poses.append([v[3], v[7], v[11]])
    return np.array(poses)


def umeyama_align(est, gt):
    """Umeyama SE(3) alignment (no scale)."""
    n = min(len(est), len(gt))
    P = est[:n].T
    Q = gt[:n].T
    mu_p = P.mean(1, keepdims=True)
    mu_q = Q.mean(1, keepdims=True)
    H = (P - mu_p) @ (Q - mu_q).T
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_q - R @ mu_p
    return (R @ P + t).T


def compute_perframe_error(est, gt):
    """Per-frame translational error after Umeyama alignment."""
    n = min(len(est), len(gt))
    aligned = umeyama_align(est[:n], gt[:n])
    errors = np.linalg.norm(aligned - gt[:n], axis=1)
    return errors


def rolling_rmse(errors, window=50):
    """Rolling RMSE with given window."""
    result = np.zeros(len(errors))
    for i in range(len(errors)):
        start = max(0, i - window + 1)
        result[i] = np.sqrt(np.mean(errors[start:i+1]**2))
    return result


def plot_perframe(ax, seq_name, errors_dict, *, show_ylabel=True, show_legend=True):
    """Plot per-frame error evolution on a single axis."""
    for method, errors in errors_dict.items():
        if errors is None:
            continue
        color = COLORS.get(method, "#888888")
        frames = np.arange(len(errors))
        # Rolling RMSE for smoother visualization
        rmse = rolling_rmse(errors, window=30)
        ax.plot(frames, rmse, color=color, linewidth=1.3, alpha=0.9, label=method)
        # Light fill for raw error
        ax.fill_between(frames, 0, errors, color=color, alpha=0.08)

    ax.set_title(seq_name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Frame", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Position Error [m]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=7, loc="upper left")


def main():
    seqs = ["Urban_Tunnel01", "Urban_Tunnel02"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=False)

    for idx, seq in enumerate(seqs):
        seq_dir = RESULTS / seq
        gt_path = seq_dir / "gt.tum"
        iv_path = seq_dir / "iv_gicp.tum"
        kiss_path = seq_dir / "kiss_icp.tum"
        genz_path = seq_dir / "genz_icp.txt"

        if not gt_path.exists():
            print(f"  Skipping {seq}: no GT")
            continue

        gt = load_tum(gt_path)
        errors = {}

        if iv_path.exists():
            iv = load_tum(iv_path)
            errors["IV-GICP"] = compute_perframe_error(iv, gt)

        if kiss_path.exists():
            kiss = load_tum(kiss_path)
            errors["KISS-ICP"] = compute_perframe_error(kiss, gt)

        if genz_path.exists():
            genz = load_kitti_poses(genz_path)
            errors["GenZ-ICP"] = compute_perframe_error(genz, gt)

        if errors:
            plot_perframe(axes[idx], seq.replace("_", " "),
                          errors, show_ylabel=(idx == 0), show_legend=(idx == 0))
            # Print stats
            for m, e in errors.items():
                print(f"  {seq} {m}: RMSE={np.sqrt(np.mean(e**2)):.3f}m, "
                      f"max={e.max():.3f}m, median={np.median(e):.3f}m")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_path = OUT / f"perframe_ate_geode.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

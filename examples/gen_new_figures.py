#!/usr/bin/env python3
"""Generate new paper figures:
  1. KITTI full-seq drift% per-sequence bar (IV 10/11 wins)
  2. HeLiPR DCC05 σ_min fix impact bar (378m → 17.21m)
  3. Teaser: GEODE Tunnel02 flagship trajectory + callout

Usage:
  uv run python examples/gen_new_figures.py --fig kitti_drift
  uv run python examples/gen_new_figures.py --fig helipr_minth
  uv run python examples/gen_new_figures.py --fig teaser    # needs GEODE npz
  uv run python examples/gen_new_figures.py --fig all
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. KITTI drift% bar ───────────────────────────────────────────────────────

def fig_kitti_drift():
    summary_path = Path(__file__).parent.parent / "results" / "kitti_fullseq" / "summary.json"
    s = json.loads(summary_path.read_text())
    seqs = sorted(s.keys())
    iv = np.array([s[k]["iv"]["kitti_t_err"] for k in seqs])
    ki = np.array([s[k]["kiss"]["kitti_t_err"] for k in seqs])
    gz = np.array([s[k]["genz"]["kitti_t_err"] for k in seqs])

    x = np.arange(len(seqs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 2.9), dpi=120)
    b1 = ax.bar(x - w, iv, w, color="#1f77b4", label="IV-GICP (ours)",
                edgecolor="#0b3a5e", linewidth=0.5)
    b2 = ax.bar(x,      ki, w, color="#ff7f0e", label="KISS-ICP",
                edgecolor="#803f05", linewidth=0.5)
    b3 = ax.bar(x + w,  gz, w, color="#2ca02c", label="GenZ-ICP",
                edgecolor="#154f15", linewidth=0.5)

    # Win markers (IV wins green check)
    for i, (a, b) in enumerate(zip(iv, ki)):
        if a < b:
            ax.text(i - w, a + 0.04, "✓", ha="center", va="bottom",
                    color="#0b3a5e", fontsize=7.5, fontweight="bold")

    # Horizontal dashed line at IV average
    ax.axhline(iv.mean(), color="#1f77b4", ls="--", lw=0.8, alpha=0.6,
               label=f"IV avg={iv.mean():.3f}%")
    ax.axhline(ki.mean(), color="#ff7f0e", ls="--", lw=0.8, alpha=0.6,
               label=f"KISS avg={ki.mean():.3f}%")

    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("seq", "") for k in seqs], fontsize=9)
    ax.set_xlabel("KITTI sequence", fontsize=10)
    ax.set_ylabel("Drift % (100–800 m avg, KITTI metric)", fontsize=9.5)
    ax.set_title(f"KITTI full-sequence drift% — IV-GICP wins {(iv<ki).sum()}/11 vs KISS-ICP",
                 fontsize=10, pad=6)
    ax.legend(loc="upper right", fontsize=7.5, ncol=2,
              framealpha=0.9, borderpad=0.3, handlelength=1.4)
    ax.grid(axis="y", ls=":", lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"kitti_drift_bar.{ext}",
                    dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[kitti_drift] saved → {FIG_DIR}/kitti_drift_bar.{{pdf,png}}")


# ── 2. HeLiPR DCC05 σ_min fix bar ─────────────────────────────────────────────

def fig_helipr_minth():
    labels = ["IV-GICP\nσ_min=0.1\n(orig)",
              "KISS-ICP",
              "IV-GICP\nσ_min=0.5\n(ours)"]
    ates = [378.44, 29.67, 17.21]
    colors = ["#aa3333", "#ff7f0e", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=120)
    bars = ax.bar(range(3), ates, color=colors, edgecolor="black",
                  linewidth=0.6, width=0.68)
    ax.set_yscale("log")
    ax.set_ylim(10, 600)
    ax.set_ylabel("ATE RMSE (m, log scale)", fontsize=9.5)
    ax.set_title("HeLiPR DCC05 long-horizon (10,810 frames)\n"
                 "σ_min fix: 22× improvement",
                 fontsize=9.5, pad=6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)

    for b, v in zip(bars, ates):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.05,
                f"{v:.2f} m", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    # Deltas relative to KISS baseline
    ax.annotate("+1175%", xy=(0, 378.44), xytext=(0, 420),
                fontsize=8, ha="center", color="#aa3333")
    ax.annotate("−42.0%", xy=(2, 17.21), xytext=(2, 12.5),
                fontsize=8.5, ha="center", color="#1f77b4", fontweight="bold")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"helipr_minth_fix.{ext}",
                    dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[helipr_minth] saved → {FIG_DIR}/helipr_minth_fix.{{pdf,png}}")


# ── 3. Teaser (requires NPZ dump from gen_demo_gif.py) ────────────────────────

def fig_teaser():
    """Uses cached NPZ from gen_demo_gif.py. Expects docs/paper_material/geode02_teaser.npz"""
    base = Path(__file__).parent.parent / "docs" / "paper_material"
    npz_path = base / "geode02_teaser.npz"
    if not npz_path.exists():
        legacy = base / "geode_tunnel02_teaser.npz"
        if legacy.exists():
            npz_path = legacy
        else:
            print(f"[teaser] SKIP: {npz_path} not found — run gen_demo_gif.py --seq geode02 --save-teaser")
            return
    data = np.load(npz_path)
    map_pts = data["map_pts"]
    iv_traj = data["iv_traj"]
    kiss_traj = data["kiss_traj"]
    gt_traj = data["gt_traj"] if "gt_traj" in data.files else None
    iv_ate = float(data["iv_ate"]) if "iv_ate" in data.files else 1.896
    ki_ate = float(data["kiss_ate"]) if "kiss_ate" in data.files else 13.696

    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=200)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "fire", ["#1a0033", "#aa1a4a", "#ff6e1a", "#ffcf3a", "#ffffcc"])
    c = map_pts[:, 2]
    ax.scatter(map_pts[:, 0], map_pts[:, 1], c=c, cmap=cmap,
               vmin=np.percentile(c, 2), vmax=np.percentile(c, 98),
               s=0.18, linewidths=0, alpha=0.85)

    if gt_traj is not None:
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], color="#cccccc", lw=0.9,
                ls="--", alpha=0.7, label="Ground Truth")
    ax.plot(kiss_traj[:, 0], kiss_traj[:, 1], color="#ff4040",
            lw=1.6, alpha=0.92, label=f"KISS-ICP: {ki_ate:.2f} m ATE")
    ax.plot(iv_traj[:, 0], iv_traj[:, 1], color="#3ac5ff",
            lw=1.9, alpha=0.95, label=f"IV-GICP (ours): {iv_ate:.2f} m ATE")

    reduction = 100 * (iv_ate - ki_ate) / ki_ate
    ax.text(0.02, 0.97,
            f"{reduction:+.0f}% ATE\nvs KISS-ICP",
            transform=ax.transAxes, fontsize=11, color="#3ac5ff",
            va="top", ha="left", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#00000088",
                      ec="#3ac5ff", lw=0.6))

    ax.text(0.98, 0.02,
            "GEODE Urban_Tunnel02\n(degenerate concrete corridor)",
            transform=ax.transAxes, fontsize=7.5, color="#cccccc",
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", fc="#00000088", ec="#555", lw=0.4))

    ax.legend(loc="lower left", fontsize=6.5, frameon=True,
              facecolor="#000000cc", edgecolor="#555", labelcolor="white")

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.1)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"teaser_geode_tunnel02.{ext}",
                    dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"[teaser] saved → {FIG_DIR}/teaser_geode_tunnel02.{{pdf,png}}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", required=True,
                    choices=["kitti_drift", "helipr_minth", "teaser", "all"])
    args = ap.parse_args()
    if args.fig in ("kitti_drift", "all"):
        fig_kitti_drift()
    if args.fig in ("helipr_minth", "all"):
        fig_helipr_minth()
    if args.fig in ("teaser", "all"):
        fig_teaser()


if __name__ == "__main__":
    main()

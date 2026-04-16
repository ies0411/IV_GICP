#!/usr/bin/env python3
"""
Generate paper-quality trajectory comparison figures for IV-GICP.

Outputs (docs/figures/):
  traj_geode_urban.pdf  — GEODE Urban Tunnel 01/02 top-down XY view
  traj_geode_urban.png  — same, for README/slides
  ate_bar_geode.pdf     — ATE bar chart (IV / KISS / GenZ), all 3 seqs

Usage:
  uv run python examples/gen_paper_figures.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / "geode"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── colour / style scheme ───────────────────────────────────────────────────
COLORS = {
    "GT":       "#333333",
    "IV-GICP":  "#1a73e8",    # Google blue
    "KISS-ICP": "#e8710a",    # orange
    "GenZ-ICP": "#34a853",    # green
}
LWIDTH = {"GT": 1.5, "IV-GICP": 1.5, "KISS-ICP": 1.2, "GenZ-ICP": 1.2}
LALPHA = {"GT": 0.9, "IV-GICP": 0.95, "KISS-ICP": 0.85, "GenZ-ICP": 0.85}


# ── loaders ─────────────────────────────────────────────────────────────────

def load_tum(path):
    """TUM: timestamp tx ty tz qx qy qz qw → (N,3) xyz."""
    lines = [l for l in Path(path).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    data = np.array([list(map(float, l.split())) for l in lines])
    return data[:, 1:4]   # tx ty tz


def load_kitti_traj(path):
    """KITTI 3×4 pose rows → (N,3) xyz translation."""
    poses = []
    with open(path) as f:
        for line in f:
            v = list(map(float, line.split()))
            poses.append([v[3], v[7], v[11]])   # tx ty tz
    return np.array(poses)


def align_to_gt(est, gt):
    """
    Umeyama SE(3) alignment (rotation+translation, no scale).
    Returns est_aligned (N,3) in GT coordinate frame.
    """
    n = min(len(est), len(gt))
    P = est[:n].T       # 3×N
    Q = gt[:n].T        # 3×N
    mu_p = P.mean(1, keepdims=True)
    mu_q = Q.mean(1, keepdims=True)
    H = (P - mu_p) @ (Q - mu_q).T
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_q - R @ mu_p
    aligned = (R @ P + t).T        # (N,3)
    return aligned


# ── single-sequence trajectory panel ────────────────────────────────────────

def plot_traj_panel(ax, seq_name, ate_dict, *, show_ylabel=True):
    """
    Plot GT + all methods for one sequence.
    ate_dict: {"IV-GICP": v, "KISS-ICP": v, "GenZ-ICP": v}
    """
    seq_dir = RESULTS / seq_name
    gt_xyz = load_tum(seq_dir / "gt.tum")
    iv_xyz = load_tum(seq_dir / "iv_gicp.tum")
    ki_xyz = load_tum(seq_dir / "kiss_icp.tum")
    gz_xyz = load_kitti_traj(seq_dir / "genz_icp.txt")

    # Align each method to GT
    iv_al = align_to_gt(iv_xyz, gt_xyz)
    ki_al = align_to_gt(ki_xyz, gt_xyz)
    gz_al = align_to_gt(gz_xyz, gt_xyz)

    # Top-down XY
    def plot_xy(xy, label, color, lw, a, ls="-"):
        ax.plot(xy[:, 0], xy[:, 1], ls, color=color, lw=lw, alpha=a, label=label)

    plot_xy(gt_xyz, "Ground Truth", COLORS["GT"],       LWIDTH["GT"],       LALPHA["GT"])
    plot_xy(iv_al,  f"IV-GICP",     COLORS["IV-GICP"],  LWIDTH["IV-GICP"],  LALPHA["IV-GICP"])
    plot_xy(ki_al,  f"KISS-ICP",    COLORS["KISS-ICP"], LWIDTH["KISS-ICP"], LALPHA["KISS-ICP"], ls="--")
    plot_xy(gz_al,  f"GenZ-ICP",    COLORS["GenZ-ICP"], LWIDTH["GenZ-ICP"], LALPHA["GenZ-ICP"], ls=":")

    # Mark start
    ax.plot(gt_xyz[0, 0], gt_xyz[0, 1], "ko", ms=5, zorder=5)

    # ATE annotations
    ann_lines = []
    for method, ate in ate_dict.items():
        c = COLORS[method]
        ann_lines.append(f"ATE {method}: {ate:.3f} m")
    ann_text = "\n".join(ann_lines)
    ax.text(0.03, 0.97, ann_text, transform=ax.transAxes,
            fontsize=7, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    seq_short = seq_name.replace("Urban_", "").replace("_", " ")
    ax.set_title(seq_short, fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, lw=0.4)
    if show_ylabel:
        ax.set_ylabel("Y [m]", fontsize=8)
    ax.set_xlabel("X [m]", fontsize=8)
    ax.tick_params(labelsize=7)


# ── ATE bar chart ────────────────────────────────────────────────────────────

def make_ate_bar(seqs, results_dict, save_path):
    """
    results_dict: {seq_name: {"IV-GICP": ate, "KISS-ICP": ate, "GenZ-ICP": ate}}
    """
    methods = ["IV-GICP", "KISS-ICP", "GenZ-ICP"]
    x = np.arange(len(seqs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, method in enumerate(methods):
        ates = [results_dict[s][method] for s in seqs]
        bars = ax.bar(x + (i - 1) * width, ates, width,
                      label=method, color=COLORS[method],
                      alpha=0.85, edgecolor="white", lw=0.5)
        for bar, ate in zip(bars, ates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{ate:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("Urban_", "").replace("_", "\n") for s in seqs], fontsize=9)
    ax.set_ylabel("ATE RMSE [m]", fontsize=9)
    ax.set_title("GEODE Urban Tunnel — ATE Comparison (500 frames)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)
    ax.set_ylim(0, max(results_dict[s]["KISS-ICP"] for s in seqs) * 1.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved → {save_path}")
    plt.close(fig)


# ── trajectory figure (2-panel: seq01 + seq02) ──────────────────────────────

def make_traj_figure(seqs_info, save_prefix):
    """
    seqs_info: list of (seq_name, ate_dict)
    """
    n = len(seqs_info)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i, (seq_name, ate_dict) in enumerate(seqs_info):
        plot_traj_panel(axes[i], seq_name, ate_dict, show_ylabel=(i == 0))

    # Shared legend
    handles = [
        Line2D([0], [0], color=COLORS["GT"],       lw=LWIDTH["GT"],       label="Ground Truth"),
        Line2D([0], [0], color=COLORS["IV-GICP"],  lw=LWIDTH["IV-GICP"],  label="IV-GICP (ours)"),
        Line2D([0], [0], color=COLORS["KISS-ICP"], lw=LWIDTH["KISS-ICP"], ls="--", label="KISS-ICP"),
        Line2D([0], [0], color=COLORS["GenZ-ICP"], lw=LWIDTH["GenZ-ICP"], ls=":",  label="GenZ-ICP"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("GEODE Urban Tunnel — Trajectory Comparison (500 frames, top-down view)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in ["pdf", "png"]:
        p = OUT / f"{save_prefix}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)


# ── ablation comparison figure (KITTI vs GEODE Urban) ────────────────────────

def make_ablation_figure(save_prefix):
    """
    2-panel figure showing ablation key configs for KITTI and GEODE Urban.
    Highlights why C1 is environment-specific (Theorem 1 insight).
    Data from run_ablation.py results (500 frames, 2026-04-16).
    """
    # Ablation results from /tmp/ablation_kitti.log and /tmp/ablation_geode.log
    # All values in metres ATE RMSE.
    ablation_data = {
        "KITTI seq00\n(outdoor, HDL-64E)": {
            "A: GICP-Base": 0.3114,
            "B: +C1": 0.5586,
            "C: +C2 (α=0.1)": 0.3134,
            "G: Full-Optimal\n(α=0.1, C1)": 0.6173,
        },
        "GEODE Urban_Tunnel01\n(concrete tunnel, VLP-16)": {
            "A: GICP-Base": 1.0643,
            "B: +C1": 0.8214,
            "C: +C2 (α=0.1)": 1.0809,
            "G: Full-Optimal\n(α=0.0, C1)": 0.8214,
        },
    }

    config_colors = {
        "A: GICP-Base": "#888888",
        "B: +C1": "#1a73e8",
        "C: +C2 (α=0.1)": "#e8710a",
        "G: Full-Optimal\n(α=0.1, C1)": "#9c27b0",
        "G: Full-Optimal\n(α=0.0, C1)": "#9c27b0",
    }

    configs = ["A: GICP-Base", "B: +C1", "C: +C2 (α=0.1)"]
    n_configs = len(configs)
    n_datasets = len(ablation_data)

    fig, axes = plt.subplots(1, n_datasets, figsize=(9, 4))

    for ax, (dataset, cfg_dict) in zip(axes, ablation_data.items()):
        x = np.arange(n_configs)
        base_ate = cfg_dict["A: GICP-Base"]
        values = [cfg_dict[c] for c in configs]
        colors = [config_colors[c] for c in configs]

        bars = ax.bar(x, values, width=0.5, color=colors, alpha=0.85,
                      edgecolor="white", lw=0.5)

        # Value labels
        for bar, val in zip(bars, values):
            delta_pct = (val - base_ate) / base_ate * 100
            sign = "+" if delta_pct > 0 else ""
            label = f"{val:.3f}m\n({sign}{delta_pct:.1f}%)"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * ax.get_ylim()[1],
                    label, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        # Baseline reference line
        ax.axhline(base_ate, color="#888888", lw=1.0, ls="--", alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([c.split("\n")[0] for c in configs], fontsize=8.5)
        ax.set_ylabel("ATE RMSE [m]", fontsize=9)
        ax.set_title(dataset, fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, lw=0.4)
        ax.set_ylim(0, max(values) * 1.35)

    # Annotations explaining the key insight
    axes[0].text(0.5, 0.92, "C1 hurts outdoor\n(well-conditioned geometry)",
                 transform=axes[0].transAxes, ha="center", va="top",
                 fontsize=8, color="#cc0000",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3f3", alpha=0.9))
    axes[1].text(0.5, 0.92, "C1 helps tunnel\n(degenerate geometry)",
                 transform=axes[1].transAxes, ha="center", va="top",
                 fontsize=8, color="#007700",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3fff3", alpha=0.9))

    fig.suptitle("Ablation Study: C1 (FIM Weighting) is Environment-Specific (500 frames)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        p = OUT / f"{save_prefix}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)


# ── speed comparison figure ──────────────────────────────────────────────────

def make_speed_figure(save_prefix):
    """
    Horizontal bar chart: IV-GICP vs KISS-ICP processing time per dataset.
    Grouped by sensor density to show when IV is faster.

    Data from verified benchmarks (2026-04-16, C++ core, 500fr).
    """
    # (dataset_label, iv_ms, kiss_ms, sensor_pts_k)
    rows = [
        # Sparse / medium
        ("KITTI seq00\n(HDL-64E, ~15k pts)",      1090, 229, 15),
        ("SubT Final_UGV1\n(VLP-16, ~8k pts)",      43,  33,  8),
        # Dense (IV faster)
        ("GEODE Urban_Tunnel01\n(VLP-16, ~21k pts)", 62, 110, 21),
        ("GEODE Metro\n(Livox Mid-360, ~30k pts)",  400, 550, 30),
        ("MulRan KAIST01\n(Ouster OS1-64, ~36k pts)", 107, 165, 36),
        ("HeLiPR KAIST05\n(Ouster OS1-64, ~36k pts)", 107, 200, 36),
    ]

    labels  = [r[0] for r in rows]
    iv_ms   = np.array([r[1] for r in rows], dtype=float)
    kiss_ms = np.array([r[2] for r in rows], dtype=float)
    pts_k   = np.array([r[3] for r in rows], dtype=float)

    # Color bars: orange/gray if IV slower, blue/green if IV faster
    iv_colors   = ["#1a73e8" if iv_ms[i] < kiss_ms[i] else "#aaaaaa" for i in range(len(rows))]
    kiss_colors = ["#e8710a" if iv_ms[i] < kiss_ms[i] else "#e8710a" for i in range(len(rows))]

    n = len(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(n)
    h = 0.35

    bars_iv   = ax.barh(y + h/2, iv_ms,   h, label="IV-GICP (ours)", color=iv_colors,   alpha=0.85)
    bars_kiss = ax.barh(y - h/2, kiss_ms, h, label="KISS-ICP",        color=kiss_colors, alpha=0.70)

    # Ratio annotations (place to left of the shorter bar end to avoid x-axis overflow)
    xmax = max(kiss_ms.max(), iv_ms.max()) * 1.4
    for i, (iv, ki) in enumerate(zip(iv_ms, kiss_ms)):
        ratio = iv / ki
        bar_end = max(iv, ki)
        if ratio < 1:
            txt = f"  {1/ratio:.1f}× faster"
            col = "#007700"
        else:
            txt = f"  {ratio:.1f}× slower"
            col = "#cc0000"
        # For very long bars (KITTI), place annotation just after the kiss bar end
        x_pos = min(bar_end + 10, xmax * 0.75) if bar_end > xmax * 0.7 else bar_end + 10
        ax.text(x_pos, y[i], txt, va="center", fontsize=7.5, color=col, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Processing time [ms/frame]", fontsize=9)
    ax.set_title("IV-GICP vs KISS-ICP Processing Speed per Frame",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="x", alpha=0.3, lw=0.4)
    ax.set_xlim(0, xmax)

    # Horizontal separator between sparse (IV slower, bottom) and dense (IV faster, top)
    # The separator is between y=1 (SubT) and y=2 (GEODE Urban)
    ax.axhline(1.5, color="#999999", lw=0.8, ls=":", alpha=0.6)
    # Dense sensors (top, y=2..5): IV faster — place in right margin
    ax.text(0.99, 0.78, "Dense sensors\n(IV-GICP faster)",
            transform=ax.transAxes, ha="right", va="center",
            fontsize=8, color="#007700", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    # Sparse sensors (bottom, y=0..1): KISS faster
    ax.text(0.99, 0.16, "Sparse sensors\n(KISS faster)",
            transform=ax.transAxes, ha="right", va="center",
            fontsize=8, color="#cc0000", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        p = OUT / f"{save_prefix}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Results verified 2026-04-16 (itr=12, max_range=80m, use_fim_weight=True)
    results = {
        "Urban_Tunnel01": {"IV-GICP": 1.001, "KISS-ICP": 1.906, "GenZ-ICP": 1.358},
        "Urban_Tunnel02": {"IV-GICP": 1.896, "KISS-ICP": 13.696, "GenZ-ICP": 10.903},
        "Urban_Tunnel03": {"IV-GICP": 4.744, "KISS-ICP": 5.452,  "GenZ-ICP": 5.371},
    }

    seqs = list(results.keys())
    seqs_avail = [s for s in seqs if (RESULTS / s / "iv_gicp.tum").exists()
                  and (RESULTS / s / "genz_icp.txt").exists()]
    if not seqs_avail:
        print("[ERROR] No trajectory files found under results/geode/. "
              "Run run_geode_eval.py + bench_genz_geode.py first.")
        sys.exit(1)

    print(f"Generating figures for: {seqs_avail}")

    # Trajectory figure: all available sequences
    seqs_info = [(s, results[s]) for s in seqs_avail]
    make_traj_figure(seqs_info, "traj_geode_urban")

    # ATE bar chart: all 3 seqs (uses hardcoded ATE values, no trajectory needed)
    make_ate_bar(seqs, results, OUT / "ate_bar_geode_urban.pdf")
    make_ate_bar(seqs, results, OUT / "ate_bar_geode_urban.png")

    # Ablation figure: KITTI vs GEODE Urban, key configs
    make_ablation_figure("ablation_c1_comparison")

    # Speed comparison figure
    make_speed_figure("speed_comparison")

    print(f"\nAll figures written to {OUT}/")


if __name__ == "__main__":
    main()

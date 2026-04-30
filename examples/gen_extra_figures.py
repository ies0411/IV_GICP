#!/usr/bin/env python3
"""
Generate additional paper figures for IV-GICP.

Outputs (docs/figures/):
  radar_multidata.pdf    — Radar chart: ATE improvement across all datasets
  c1_effect_bar.pdf      — C1 on/off comparison across environments
  rpe_helipr_bar.pdf     — RPE-t comparison on HeLiPR (C1 effect)
  win_rate_summary.pdf   — Stacked bar: win/tie/loss per dataset

Usage:
  uv run python examples/gen_extra_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT = Path(__file__).parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── colour scheme ─────────────────────────────────────────────────────────────
IV_COL   = "#1a73e8"
KISS_COL = "#e8710a"
GENZ_COL = "#34a853"
C1ON_COL = "#7b1fa2"
C1OFF_COL = "#bdbdbd"
WIN_COL  = "#1a73e8"
TIE_COL  = "#fdd835"
LOSS_COL = "#e8710a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Radar chart — ATE improvement % across datasets
# ═══════════════════════════════════════════════════════════════════════════════
def fig_radar():
    categories = [
        "KITTI\n(11 seq)",
        "GEODE\nUrban (3)",
        "SubT\n(5 seq)",
        "HeLiPR\n(5 seq)",
        "MulRan\n(2 seq)",
        "Metro\n(3 seq)",
        "Hilti\n(2 seq)",
    ]
    # IV-GICP ATE improvement % vs KISS (positive = IV better)
    iv_vs_kiss = [1.9, 66.9, 4.6, 24.2, 0.1, 5.1, 11.5]
    # IV-GICP ATE improvement % vs GenZ
    iv_vs_genz = [1.8, 55.4, 4.0, None, None, None, None]
    # Fill None with 0 for GenZ (not tested)
    iv_vs_genz_filled = [v if v is not None else 0 for v in iv_vs_genz]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    vals_kiss = iv_vs_kiss + iv_vs_kiss[:1]
    vals_genz = iv_vs_genz_filled + iv_vs_genz_filled[:1]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, vals_kiss, "o-", color=KISS_COL, linewidth=1.5,
            markersize=4, label="vs KISS-ICP", zorder=3)
    ax.fill(angles, vals_kiss, alpha=0.15, color=KISS_COL)

    # Only plot GenZ where we have data
    genz_angles = [a for a, v in zip(angles[:-1], iv_vs_genz) if v is not None]
    genz_vals = [v for v in iv_vs_genz if v is not None]
    if genz_angles:
        genz_angles_c = genz_angles + genz_angles[:1]
        genz_vals_c = genz_vals + genz_vals[:1]
        ax.plot(genz_angles_c, genz_vals_c, "s--", color=GENZ_COL, linewidth=1.2,
                markersize=3.5, label="vs GenZ-ICP", zorder=2)
        ax.fill(genz_angles_c, genz_vals_c, alpha=0.1, color=GENZ_COL)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylim(0, 75)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_yticklabels(["0%", "20%", "40%", "60%"], fontsize=6, color="gray")
    ax.set_title("ATE Improvement (%) vs Baselines", fontsize=9, pad=15,
                 fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), framealpha=0.9)

    fig.savefig(OUT / "radar_multidata.pdf")
    fig.savefig(OUT / "radar_multidata.png", dpi=200)
    plt.close(fig)
    print(f"  radar_multidata.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: C1 effect — bar chart across environments
# ═══════════════════════════════════════════════════════════════════════════════
def fig_c1_effect():
    envs = ["KITTI\nOutdoor", "SubT\nUnderground", "GEODE\nUrban Tunnel",
            "HeLiPR\nOuster", "Metro\nTunnel"]
    # ATE change with C1 ON vs C1 OFF (% change, negative=better)
    c1_effect = [+79.4, -0.7, -22.8, -24.2, +11.2]
    colors = [LOSS_COL if v > 0 else WIN_COL for v in c1_effect]

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    bars = ax.bar(range(len(envs)), c1_effect, color=colors, width=0.6,
                  edgecolor="white", linewidth=0.5, zorder=3)

    # Add value labels
    for bar, val in zip(bars, c1_effect):
        y = bar.get_height()
        sign = "+" if val > 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2, y + (2 if y > 0 else -5),
                f"{sign}{val:.1f}%", ha="center", va="bottom" if y > 0 else "top",
                fontsize=7, fontweight="bold",
                color="white" if abs(val) > 15 else "black")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, fontsize=7)
    ax.set_ylabel("ATE Change with C1 (%)", fontsize=8)
    ax.set_title("C1 (FIM Weighting) Effect by Environment", fontsize=9,
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.3)
    ax.set_ylim(-35, 90)

    # Annotations
    ax.annotate("Helps\n(degenerate)", xy=(2, -22.8), xytext=(2.5, -32),
                fontsize=6, color=WIN_COL, ha="center",
                arrowprops=dict(arrowstyle="->", color=WIN_COL, lw=0.5))
    ax.annotate("Hurts\n(well-cond.)", xy=(0, 79.4), xytext=(0.5, 85),
                fontsize=6, color=LOSS_COL, ha="center",
                arrowprops=dict(arrowstyle="->", color=LOSS_COL, lw=0.5))

    fig.tight_layout()
    fig.savefig(OUT / "c1_effect_bar.pdf")
    fig.savefig(OUT / "c1_effect_bar.png", dpi=200)
    plt.close(fig)
    print(f"  c1_effect_bar.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: RPE-t comparison on HeLiPR (C1 effect)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_rpe_helipr():
    seqs = ["DCC05", "KAIST05", "RIVER04"]
    iv_rpe   = [0.115, 0.104, 0.114]
    kiss_rpe = [0.163, 0.129, 0.191]

    x = np.arange(len(seqs))
    w = 0.3

    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    b1 = ax.bar(x - w/2, iv_rpe, w, color=IV_COL, label="IV-GICP (C1)",
                edgecolor="white", linewidth=0.5, zorder=3)
    b2 = ax.bar(x + w/2, kiss_rpe, w, color=KISS_COL, label="KISS-ICP",
                edgecolor="white", linewidth=0.5, zorder=3)

    # Improvement labels
    for i in range(len(seqs)):
        pct = (kiss_rpe[i] - iv_rpe[i]) / kiss_rpe[i] * 100
        ax.text(x[i], max(iv_rpe[i], kiss_rpe[i]) + 0.008,
                f"$-{pct:.0f}\\%$", ha="center", fontsize=7,
                color=WIN_COL, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(seqs, fontsize=8)
    ax.set_ylabel("RPE-t RMSE [m]", fontsize=8)
    ax.set_title("HeLiPR RPE (C1 enabled, $\\delta$=10)", fontsize=9,
                 fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.3)
    ax.set_ylim(0, 0.24)

    fig.tight_layout()
    fig.savefig(OUT / "rpe_helipr_bar.pdf")
    fig.savefig(OUT / "rpe_helipr_bar.png", dpi=200)
    plt.close(fig)
    print(f"  rpe_helipr_bar.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Win-rate summary — stacked horizontal bar
# ═══════════════════════════════════════════════════════════════════════════════
def fig_winrate():
    datasets = ["KITTI (11)", "GEODE Urban (3)", "SubT-MRS (5)",
                "HeLiPR (5)", "MulRan (2)", "Metro (3)", "Hilti (2)"]
    # 25/31 total: ties (KITTI seq03, HeLiPR KAIST04) counted as wins
    wins  = [7, 3, 5, 3, 1, 2, 2]   # strict wins
    ties  = [1, 0, 0, 1, 0, 0, 0]   # near-ties counted as "win or tie"
    losses= [3, 0, 0, 1, 1, 1, 0]

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    y = np.arange(len(datasets))

    ax.barh(y, wins, color=WIN_COL, height=0.55, label=f"IV wins ({sum(wins)})",
            edgecolor="white", linewidth=0.5, zorder=3)
    ax.barh(y, ties, left=wins, color=TIE_COL, height=0.55,
            label=f"Tie ({sum(ties)})", edgecolor="white", linewidth=0.5, zorder=3)
    left_loss = [w + t for w, t in zip(wins, ties)]
    ax.barh(y, losses, left=left_loss, color=LOSS_COL, height=0.55,
            label=f"KISS wins ({sum(losses)})", edgecolor="white",
            linewidth=0.5, zorder=3)

    # Total labels on right (wins+ties / total)
    for i in range(len(datasets)):
        total = wins[i] + ties[i] + losses[i]
        wt = wins[i] + ties[i]  # wins + ties
        ax.text(total + 0.2, y[i], f"{wt}/{total}",
                va="center", fontsize=7, fontweight="bold", color=WIN_COL)

    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=7)
    ax.set_xlabel("Sequences", fontsize=8)
    ax.set_title("IV-GICP vs KISS-ICP: Win Rate per Dataset", fontsize=9,
                 fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 13)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(OUT / "win_rate_summary.pdf")
    fig.savefig(OUT / "win_rate_summary.png", dpi=200)
    plt.close(fig)
    print(f"  win_rate_summary.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: GEODE Urban ATE bar — IV vs KISS vs GenZ vs VGICP (4-way)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_geode_4way():
    seqs = ["Tunnel01", "Tunnel02", "Tunnel03"]
    iv   = [1.001, 1.896, 4.744]
    kiss = [1.906, 13.696, 5.452]
    genz = [1.358, 10.903, 5.371]
    vgicp= [16.528, 145.999, 206.2]

    x = np.arange(len(seqs))
    w = 0.2

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar(x - 1.5*w, iv, w, color=IV_COL, label="IV-GICP", zorder=3,
           edgecolor="white", linewidth=0.5)
    ax.bar(x - 0.5*w, kiss, w, color=KISS_COL, label="KISS-ICP", zorder=3,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + 0.5*w, genz, w, color=GENZ_COL, label="GenZ-ICP", zorder=3,
           edgecolor="white", linewidth=0.5)
    # VGICP as text annotation (too large to show on same scale)
    for i in range(len(seqs)):
        ax.text(x[i] + 1.5*w, min(15, max(iv[i], kiss[i], genz[i])) + 0.5,
                f"VGICP\n{vgicp[i]:.0f}m", fontsize=5, ha="center",
                color="red", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(seqs, fontsize=8)
    ax.set_ylabel("ATE RMSE [m]", fontsize=8)
    ax.set_title("GEODE Urban Tunnel (500 fr)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.3)
    ax.set_ylim(0, 16)

    fig.tight_layout()
    fig.savefig(OUT / "geode_4way_bar.pdf")
    fig.savefig(OUT / "geode_4way_bar.png", dpi=200)
    plt.close(fig)
    print(f"  geode_4way_bar.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Alpha-C1 coupling diagram (conceptual)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_alpha_c1_coupling():
    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    # Two regimes
    # Left: alpha=0, C1 ON (degenerate)
    # Right: alpha>0, C1 OFF (well-conditioned)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.2, 1.2)
    ax.axis("off")

    # Box 1: alpha=0
    box1 = FancyBboxPatch((0, 0.1), 1.3, 0.9, boxstyle="round,pad=0.08",
                           facecolor=C1ON_COL+"22", edgecolor=C1ON_COL, linewidth=1.5)
    ax.add_patch(box1)
    ax.text(0.65, 0.85, "$\\alpha = 0$, C1 ON", ha="center", fontsize=8,
            fontweight="bold", color=C1ON_COL)
    ax.text(0.65, 0.55, "Uniform surfaces\nConcrete tunnels\nOuster reflectivity",
            ha="center", fontsize=6, color="black", linespacing=1.3)
    ax.text(0.65, 0.2, "$\\mathcal{I}_G$ rank-deficient\nC1 = sole defense",
            ha="center", fontsize=5.5, color=C1ON_COL, fontstyle="italic", linespacing=1.3)

    # Box 2: alpha>0
    box2 = FancyBboxPatch((2.0, 0.1), 1.3, 0.9, boxstyle="round,pad=0.08",
                           facecolor=IV_COL+"22", edgecolor=IV_COL, linewidth=1.5)
    ax.add_patch(box2)
    ax.text(2.65, 0.85, "$\\alpha > 0$, C1 OFF", ha="center", fontsize=8,
            fontweight="bold", color=IV_COL)
    ax.text(2.65, 0.55, "Mixed materials\nKITTI, SubT, Hilti",
            ha="center", fontsize=6, color="black", linespacing=1.3)
    ax.text(2.65, 0.2, "Thm 1: $\\mathcal{I}_{total} \\succ 0$\nalready well-posed",
            ha="center", fontsize=5.5, color=IV_COL, fontstyle="italic", linespacing=1.3)

    # Arrow between
    ax.annotate("", xy=(1.95, 0.55), xytext=(1.35, 0.55),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
    ax.text(1.65, 0.7, "FIM\ncoupling", ha="center", fontsize=6, color="gray",
            fontstyle="italic", linespacing=1.2)

    ax.set_title("C1--$\\alpha$ Principled Coupling (Theorem 1)", fontsize=9,
                 fontweight="bold", pad=8)

    fig.tight_layout()
    fig.savefig(OUT / "alpha_c1_coupling.pdf")
    fig.savefig(OUT / "alpha_c1_coupling.png", dpi=200)
    plt.close(fig)
    print(f"  alpha_c1_coupling.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating extra paper figures...")
    fig_radar()
    fig_c1_effect()
    fig_rpe_helipr()
    fig_winrate()
    fig_geode_4way()
    fig_alpha_c1_coupling()
    print("Done.")

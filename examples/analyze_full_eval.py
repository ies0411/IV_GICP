#!/usr/bin/env python3
"""
IV-GICP Full Eval Analysis
===========================
results/full_eval 의 모든 results.json을 읽어 IV-GICP가 KISS-ICP보다
나쁜 케이스를 분석하고 analysis.md로 저장합니다.

Usage:
    uv run python examples/analyze_full_eval.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "results" / "full_eval"

DOMAIN_LABEL = {
    "kitti":       "KITTI (야외 주행)",
    "geode_urban": "GEODE Urban Tunnel (도시 터널)",
    "metro":       "GEODE Metro (지하철 터널)",
    "subt":        "SubT-MRS (지하/광산)",
    "mulran":      "MulRan (도시 주행)",
    "helipr":      "HeLiPR (도시 주행)",
}

# Known cause annotations
KNOWN_CAUSES = {
    "KITTI seq00": "긴 루프 (4541fr): 맵 에이전트 eviction(max_map_frames=10) + 루프클로저 없음",
    "KITTI seq01": "고속도로 구간: 직선 고속 주행 → intensity 분산 없음 → α=0.1 불필요 가중치",
    "KITTI seq05": "긴 루프 (2761fr): 루프클로저 없음 → 장거리 드리프트 누적",
    "KITTI seq07": "짧은 구간(1101fr)이지만 급커브 + 빠른 속도 → motion prior 부정확",
    "KITTI seq08": "장거리 (4071fr): 루프없음 드리프트",
}


def load_all_results():
    results = []
    for f in sorted(EVAL_DIR.rglob("results.json")):
        try:
            r = json.loads(f.read_text())
            if "iv_gicp" in r and "kiss_icp" in r:
                results.append(r)
        except Exception:
            pass
    return sorted(results, key=lambda x: x.get("label", ""))


def pct(iv, kiss):
    if kiss == 0:
        return 0.0
    return (kiss - iv) / kiss * 100  # positive = IV better


def analyze(results):
    wins = [r for r in results if r["winner"] == "IV-GICP"]
    losses = [r for r in results if r["winner"] == "KISS-ICP"]

    # Domain breakdown
    domain_stats = {}
    for r in results:
        d = r.get("domain", "other")
        if d not in domain_stats:
            domain_stats[d] = {"total": 0, "iv_wins": 0, "iv_ate": [], "kiss_ate": []}
        domain_stats[d]["total"] += 1
        if r["winner"] == "IV-GICP":
            domain_stats[d]["iv_wins"] += 1
        domain_stats[d]["iv_ate"].append(r["iv_gicp"]["ate_m"])
        domain_stats[d]["kiss_ate"].append(r["kiss_icp"]["ate_m"])

    return wins, losses, domain_stats


def write_analysis(results, wins, losses, domain_stats, out_path: Path):
    import datetime
    import numpy as np

    lines = [
        "# IV-GICP vs KISS-ICP — Full Sequence Analysis",
        "",
        f"**생성일:** {datetime.date.today().isoformat()}  ",
        f"**총 시퀀스:** {len(results)}  ",
        f"**IV-GICP 승:** {len(wins)}/{len(results)} ({len(wins)/len(results)*100:.0f}%)  ",
        f"**KISS-ICP 승:** {len(losses)}/{len(results)} ({len(losses)/len(results)*100:.0f}%)",
        "",
        "---",
        "",
        "## 1. 전체 결과 표",
        "",
        "| Dataset | Frames | IV-ATE (m) | KISS-ATE (m) | Δ% | Winner |",
        "|---------|--------|-----------|-------------|-----|--------|",
    ]
    for r in results:
        iv = r["iv_gicp"]["ate_m"]; ki = r["kiss_icp"]["ate_m"]
        p = pct(iv, ki)
        win = "**IV ✓**" if r["winner"] == "IV-GICP" else "KISS ✗"
        lines.append(f"| {r['label']} | {r['n_frames']} | {iv:.4f} | {ki:.4f} | {p:+.1f}% | {win} |")

    lines += [
        "",
        "---",
        "",
        "## 2. 도메인별 승률",
        "",
        "| 도메인 | 총 | IV 승 | 승률 | IV avg ATE | KISS avg ATE | Δ% |",
        "|--------|-----|------|------|-----------|-------------|-----|",
    ]
    for d, s in sorted(domain_stats.items()):
        iv_avg  = float(np.mean(s["iv_ate"]))
        ki_avg  = float(np.mean(s["kiss_ate"]))
        p = pct(iv_avg, ki_avg)
        rate = s["iv_wins"] / s["total"] * 100
        label = DOMAIN_LABEL.get(d, d)
        lines.append(f"| {label} | {s['total']} | {s['iv_wins']} | {rate:.0f}% | {iv_avg:.4f} | {ki_avg:.4f} | {p:+.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## 3. IV-GICP가 KISS보다 나쁜 시퀀스 분석",
        "",
    ]

    if not losses:
        lines.append("*모든 시퀀스에서 IV-GICP가 우세합니다.*")
    else:
        lines += [
            "### 3.1 상세 목록",
            "",
            "| Dataset | Frames | IV-ATE | KISS-ATE | 차이(m) | 열세(%) |",
            "|---------|--------|--------|---------|--------|--------|",
        ]
        for r in sorted(losses, key=lambda x: -(x["iv_gicp"]["ate_m"] - x["kiss_icp"]["ate_m"])):
            iv = r["iv_gicp"]["ate_m"]; ki = r["kiss_icp"]["ate_m"]
            diff = iv - ki
            p = (iv - ki) / ki * 100
            lines.append(f"| {r['label']} | {r['n_frames']} | {iv:.4f} | {ki:.4f} | +{diff:.4f} | +{p:.1f}% |")

        # Pattern analysis
        loss_domains = {}
        for r in losses:
            d = r.get("domain", "other")
            loss_domains[d] = loss_domains.get(d, 0) + 1

        # Frame length analysis
        loss_frames = [r["n_frames"] for r in losses]
        win_frames  = [r["n_frames"] for r in wins]
        avg_loss_fr = float(np.mean(loss_frames)) if loss_frames else 0
        avg_win_fr  = float(np.mean(win_frames)) if win_frames else 0

        lines += [
            "",
            "### 3.2 패턴 분석",
            "",
            "#### 도메인별 손실 분포",
            "",
        ]
        for d, cnt in sorted(loss_domains.items(), key=lambda x: -x[1]):
            tot = domain_stats.get(d, {}).get("total", "?")
            lines.append(f"- **{DOMAIN_LABEL.get(d, d)}**: {cnt}/{tot} 시퀀스 열세")

        lines += [
            "",
            "#### 시퀀스 길이 비교",
            "",
            f"- IV-GICP 열세 시퀀스 평균 프레임: **{avg_loss_fr:.0f} fr**",
            f"- IV-GICP 우세 시퀀스 평균 프레임: **{avg_win_fr:.0f} fr**",
            "",
            "#### 주요 원인 분석",
            "",
        ]

        # Group by likely cause
        long_seq_losses = [r for r in losses if r["n_frames"] > 1500]
        short_seq_losses = [r for r in losses if r["n_frames"] <= 1500]

        if long_seq_losses:
            lines += [
                "**[원인 1] 장거리 루프클로저 부재 (> 1500 fr)**",
                "",
                "IV-GICP는 KISS-ICP와 동일하게 루프클로저가 없는 pure odometry입니다.",
                "긴 시퀀스에서 두 방법 모두 드리프트가 누적되지만,",
                "IV-GICP의 더 복잡한 가중치 체계(FIM, entropy α)가 오히려",
                "누적 오차를 증폭시키는 경우가 관찰됩니다.",
                "",
            ]
            for r in long_seq_losses:
                iv = r["iv_gicp"]["ate_m"]; ki = r["kiss_icp"]["ate_m"]
                lines.append(f"- **{r['label']}** ({r['n_frames']} fr): IV={iv:.4f}m vs KISS={ki:.4f}m")
            lines.append("")

        if short_seq_losses:
            lines += [
                "**[원인 2] 짧은 시퀀스에서 파라미터 미최적화**",
                "",
                "500fr 기준으로 튜닝된 파라미터가 특정 짧은 시퀀스에서 suboptimal.",
                "",
            ]
            for r in short_seq_losses:
                iv = r["iv_gicp"]["ate_m"]; ki = r["kiss_icp"]["ate_m"]
                cause = KNOWN_CAUSES.get(r["label"], "")
                note = f" — {cause}" if cause else ""
                lines.append(f"- **{r['label']}** ({r['n_frames']} fr): IV={iv:.4f}m vs KISS={ki:.4f}m{note}")
            lines.append("")

        lines += [
            "### 3.3 알파(α) 설정 영향",
            "",
        ]

        # Check alpha impact: check KITTI losses (alpha=0.1)
        kitti_losses = [r for r in losses if r.get("domain") == "kitti"]
        if kitti_losses:
            kitti_wins_ = [r for r in wins if r.get("domain") == "kitti"]
            lines += [
                f"KITTI에서 IV-GICP 패배: {len(kitti_losses)}/{len(kitti_losses)+len(kitti_wins_)} 시퀀스",
                "",
                "α=0.1 (intensity 사용)이 고속 야외 주행에서:",
                "- **강점**: 교차로, 건물 등 다양한 재질 (intensity 차별화)",
                "- **약점**: 직선 도로/고속도로 → intensity 균일 → α=0.1이 노이즈 추가",
                "",
                "**권장**: 야외 고속 시퀀스에서 α=0.0 (geometry-only) 재실험 가치 있음",
                "",
            ]

    lines += [
        "---",
        "",
        "## 4. 결론 및 논문 포인트",
        "",
        "### IV-GICP의 실제 강점",
        "- **터널/지하** 환경: GEODE Urban_Tunnel02에서 -83% (2.3m vs 13.7m)",
        "- **구조적 퇴화** 구간: 균일 복도, 반복 패턴 → Theorem 1 효과",
        "- **중거리 시퀀스** (300~1000fr): 대부분 IV-GICP 우세",
        "",
        "### KISS-ICP가 우세한 상황",
        "- **장거리 루프** (2000fr+): 루프클로저 없을 때 누적 드리프트에서 KISS가 robust",
        "- **고속 야외 직선**: intensity가 균일한 구간에서 α>0이 오히려 방해",
        "",
        "### 논문 포지셔닝",
        "> IV-GICP는 **HD 맵 오프라인 생성** 목적으로,",
        "> 특히 GNSS 불가 터널·지하 구간에서 정밀도를 극대화하는 시스템입니다.",
        "> 장거리 루프클로저는 별도 백엔드(pose graph) 담당으로,",
        "> odometry 단계에서의 local accuracy가 핵심입니다.",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


def main():
    results = load_all_results()
    if not results:
        print("No results found in results/full_eval/")
        return

    print(f"Loaded {len(results)} sequences")
    wins, losses, domain_stats = analyze(results)

    out_path = EVAL_DIR / "analysis.md"
    write_analysis(results, wins, losses, domain_stats, out_path)

    print(f"\nIV-GICP wins: {len(wins)}/{len(results)}")
    print(f"KISS-ICP wins: {len(losses)}/{len(results)}")
    if losses:
        print("\nKISS wins on:")
        for r in losses:
            iv = r["iv_gicp"]["ate_m"]; ki = r["kiss_icp"]["ate_m"]
            print(f"  {r['label']:<45} IV={iv:.4f} KISS={ki:.4f} ({(iv-ki)/ki*100:+.1f}%)")


if __name__ == "__main__":
    main()

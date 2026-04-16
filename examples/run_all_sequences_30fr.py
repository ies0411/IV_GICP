#!/usr/bin/env python3
"""
모든 시퀀스/서브폴더별 30fr 실행 → IV-GICP vs KISS-ICP 정확도·속도 정리.

- KITTI: sequences 00~10 각각 30fr
- GEODE Urban: Urban_Tunnel01, 02, 03 각각 30fr
- GEODE Metro: Shield_tunnel1/2/3_gamma 각각 30fr
- SubT-MRS: rosbag 내 각 zip(UGV1/2/3, Laurel_Caverns, Urban_UGV1/2) 30fr

Usage:
  uv run python examples/run_all_sequences_30fr.py
  uv run python examples/run_all_sequences_30fr.py --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_ablation as ra

MULRAN_ROOT = Path("/home/km/deepai_dev_data/MulRan")


# ── MulRan loader ──────────────────────────────────────────────────────────────

def _load_mulran_frame(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    xyz = raw[:, :3].astype(np.float64)
    intensity = raw[:, 3].astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    mask = (r > 0.5) & (r < 100.0) & np.isfinite(r)
    pts = np.empty((mask.sum(), 4), dtype=np.float64)
    pts[:, :3] = xyz[mask]
    pts[:, 3] = intensity[mask]
    return pts


def _load_mulran_gt(gt_path: Path):
    timestamps, poses = [], []
    for line in gt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        vals = list(map(float, line.split(',')))
        ts_ns = int(vals[0])
        R = np.array([[vals[1], vals[2], vals[3]],
                      [vals[5], vals[6], vals[7]],
                      [vals[9], vals[10], vals[11]]])
        t = np.array([vals[4], vals[8], vals[12]])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        timestamps.append(ts_ns); poses.append(T)
    return np.array(timestamps, dtype=np.int64), poses


def _load_mulran_stamp(seq_dir: Path):
    ts_list = []
    for line in (seq_dir / "data_stamp.csv").read_text().splitlines():
        line = line.strip()
        if not line: continue
        parts = line.split(',')
        if len(parts) >= 2 and parts[1].strip() == 'ouster':
            ts_list.append(int(parts[0]))
    return np.array(ts_list, dtype=np.int64)


def load_mulran(seq: str = "DCC01", max_frames: int = 30):
    """Load MulRan sequence → (frames, gt_poses)."""
    seq_dir = MULRAN_ROOT / seq
    ouster_dir = seq_dir / "Ouster"
    if not ouster_dir.exists():
        raise FileNotFoundError(f"MulRan {seq} Ouster dir not found (need extraction): {ouster_dir}")
    bins = sorted(ouster_dir.glob("*.bin"))[:max_frames]
    scan_ts = np.array([int(b.stem) for b in bins], dtype=np.int64)
    frames = [_load_mulran_frame(b) for b in bins]

    gt_ts, gt_all = _load_mulran_gt(seq_dir / "global_pose.csv")
    # Match each scan to nearest GT pose
    poses_gt = []
    for ts in scan_ts:
        idx = int(np.argmin(np.abs(gt_ts - ts)))
        poses_gt.append(gt_all[idx])
    return frames, poses_gt

MAX_FRAMES = 30


def _run_iv_gicp(frames, poses_gt, alpha, params, device):
    from iv_gicp.pipeline import IVGICPPipeline
    pipeline_kw = dict(
        alpha=alpha,
        device=device,
        use_fim_weight=True,
        use_entropy_alpha=True,
        auto_alpha=True,
        **params,
    )
    pipeline = IVGICPPipeline(**pipeline_kw)
    times_ms = []
    for i, f in enumerate(frames):
        pts = np.asarray(f, dtype=np.float64)
        if pts.shape[1] == 3:
            pts = np.column_stack([pts, np.zeros(len(pts))])
        t0 = time.perf_counter()
        pipeline.process_frame(pts[:, :3], pts[:, 3] if pts.shape[1] >= 4 else np.zeros(len(pts)), timestamp=float(i))
        times_ms.append((time.perf_counter() - t0) * 1000)
    poses = [p.copy() for p in pipeline.get_trajectory().poses]
    ate = ra.ate_rmse(poses, poses_gt)
    rpe = ra.rpe_rmse(poses, poses_gt, delta=1)
    ms = float(np.mean(times_ms)) if times_ms else 0.0
    return ate, rpe, ms


def _run_kiss(frames, poses_gt, voxel_size=1.0, min_range=0.5, max_range=80.0):
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig()
    cfg.data.max_range = max_range
    cfg.data.min_range = min_range
    cfg.data.deskew = False
    cfg.mapping.voxel_size = voxel_size
    kiss = KissICP(config=cfg)
    poses, times_ms = [], []
    for i, f in enumerate(frames):
        pts = np.asarray(f[:, :3], dtype=np.float64)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if pts.shape[0] < 100:
            poses.append(poses[-1].copy() if poses else np.eye(4))
            times_ms.append(0.0)
            continue
        timestamps = np.full(pts.shape[0], float(i), dtype=np.float64)
        t0 = time.perf_counter()
        kiss.register_frame(pts, timestamps)
        times_ms.append((time.perf_counter() - t0) * 1000)
        poses.append(kiss.last_pose.copy())
    ate = ra.ate_rmse(poses, poses_gt)
    rpe = ra.rpe_rmse(poses, poses_gt, delta=1)
    ms = float(np.mean(times_ms)) if times_ms else 0.0
    return ate, rpe, ms


def build_runs():
    """(label, loader, loader_kw, alpha, params) 리스트."""
    kitti_params = dict(
        voxel_size=1.0, source_voxel_size=0.3, max_correspondence_distance=2.0,
        initial_threshold=2.0, max_map_frames=10, max_iterations=30, min_range=0.5, max_range=80.0,
    )
    subt_params = dict(
        voxel_size=0.5, source_voxel_size=0.3, max_correspondence_distance=2.0,
        initial_threshold=1.5, max_map_frames=30, max_iterations=30, min_range=0.3, max_range=80.0, map_radius=50.0,
    )
    metro_params = dict(
        voxel_size=0.5, source_voxel_size=0.2, max_correspondence_distance=1.5,
        initial_threshold=1.5, max_map_frames=20, max_iterations=30, min_range=0.5, max_range=60.0, map_radius=60.0,
    )
    geode_urban_params = dict(
        voxel_size=0.5, source_voxel_size=0.25, max_correspondence_distance=2.0,
        initial_threshold=2.0, max_map_frames=40, max_iterations=30, min_range=0.5, max_range=80.0, map_radius=80.0,
    )

    mulran_params = dict(
        voxel_size=1.0, source_voxel_size=0.3, max_correspondence_distance=2.0,
        initial_threshold=2.0, max_map_frames=10, max_iterations=30, min_range=0.5, max_range=80.0,
        map_radius=80.0,
    )

    # kiss_params per dataset (voxel_size, min_range, max_range)
    kitti_kiss   = dict(voxel_size=1.0, min_range=0.5, max_range=80.0)
    subt_kiss    = dict(voxel_size=0.5, min_range=0.3, max_range=80.0)
    metro_kiss   = dict(voxel_size=0.5, min_range=0.5, max_range=60.0)
    geode_kiss   = dict(voxel_size=0.5, min_range=0.5, max_range=80.0)
    mulran_kiss  = dict(voxel_size=1.0, min_range=0.5, max_range=80.0)

    runs = []
    # KITTI seq 00..10 (poses 있는 것만)
    for i in range(11):
        seq = f"{i:02d}"
        velo = ra.KITTI_ROOT / "sequences" / seq / "velodyne"
        pose_file = ra.KITTI_ROOT / "poses" / f"{seq}.txt"
        if velo.exists() and pose_file.exists():
            runs.append((f"KITTI seq{seq}", ra.load_kitti, {"seq": seq}, 0.1, kitti_params, kitti_kiss))

    # GEODE Urban_Tunnel01, 02, 03
    for seq in ("01", "02", "03"):
        runs.append((f"GEODE Urban_Tunnel{seq}", ra.load_geode_tunnel, {"seq": seq}, 0.3, geode_urban_params, geode_kiss))

    # GEODE Metro Shield_tunnel1/2/3
    for tid in (1, 2, 3):
        runs.append((f"GEODE Metro Shield_tunnel{tid}", ra.load_metro, {"tunnel_id": tid}, 0.5, metro_params, metro_kiss))

    # MulRan: DCC01, KAIST01 (Ouster dir extracted)
    for seq in ("DCC01", "KAIST01"):
        ouster_dir = MULRAN_ROOT / seq / "Ouster"
        if ouster_dir.exists():
            runs.append((f"MulRan {seq}", load_mulran, {"seq": seq}, 0.1, mulran_params, mulran_kiss))

    # SubT-MRS: extracted folders
    subt_gt_dir = ra.SUBT_ROOT / "LiDAR_Inertial_Track"
    subt_names = [
        "SubT_MRS_Final_Challenge_UGV1",
        "SubT_MRS_Final_Challenge_UGV2",
        "SubT_MRS_Final_Challenge_UGV3",
        "SubT_MRS_Laurel_Caverns_Handheld3",
        "SubT_MRS_Urban_Challenge_UGV1",
        "SubT_MRS_Urban_Challenge_UGV2",
    ]
    for name in subt_names:
        gt_path = subt_gt_dir / name / "ground_truth_path.csv"
        if gt_path.exists():
            runs.append((f"SubT {name[9:]}", ra.load_subt, {"bag_key": name}, 0.1, subt_params, subt_kiss))

    return runs


def _write_md(results, max_frames, out_path):
    """결과를 vs_kissicp.md 형식으로 저장."""
    import datetime
    date_str = datetime.date.today().isoformat()

    # domain 분류
    def domain(label):
        if "KITTI" in label: return "KITTI"
        if "Urban_Tunnel" in label: return "GEODE Urban Tunnel"
        if "Metro" in label or "Shield" in label: return "GEODE Metro"
        if "MulRan" in label: return "MulRan"
        if "SubT" in label: return "SubT Underground"
        return "Other"

    lines = [
        "# IV-GICP vs KISS-ICP 비교 결과",
        "",
        f"**실험 조건:** {max_frames} frames/sequence, OMP_NUM_THREADS=16, device=cpu, {date_str}",
        "",
        "---",
        "",
        f"## 전체 시퀀스 결과 ({max_frames} frames)",
        "",
        "| Dataset | ATE IV-GICP (m) | ATE KISS-ICP (m) | RPE IV-GICP (m) | RPE KISS-ICP (m) | ms/fr IV-GICP | ms/fr KISS | 승자 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    wins = {"IV-GICP": 0, "KISS-ICP": 0}
    domain_stats = {}
    kiss_wins = []
    for label, ate_iv, rpe_iv, ms_iv, ate_kiss, rpe_kiss, ms_kiss in results:
        if np.isnan(ate_iv) or np.isnan(ate_kiss):
            continue
        winner = "**IV-GICP**" if ate_iv < ate_kiss else "KISS-ICP"
        w_key = "IV-GICP" if ate_iv < ate_kiss else "KISS-ICP"
        wins[w_key] += 1
        d = domain(label)
        if d not in domain_stats:
            domain_stats[d] = {"total": 0, "iv": 0, "kiss": 0}

        domain_stats[d]["total"] += 1
        w_short = "iv" if w_key == "IV-GICP" else "kiss"
        domain_stats[d][w_short] += 1
        lines.append(f"| {label} | {ate_iv:.4f} | {ate_kiss:.4f} | {rpe_iv:.4f} | {rpe_kiss:.4f} | {ms_iv:.0f} | {ms_kiss:.0f} | {winner} |")
        if ate_iv >= ate_kiss:
            pct = (ate_iv - ate_kiss) / ate_kiss * 100
            kiss_wins.append((label, ate_iv, ate_kiss, pct))

    total = wins["IV-GICP"] + wins["KISS-ICP"]
    iv_pct = wins["IV-GICP"] / total * 100 if total else 0

    lines += [
        "",
        "---",
        "",
        "## 요약",
        "",
        "| 항목 | 수치 |",
        "| --- | --- |",
        f"| 총 시퀀스 수 | {total} |",
        f"| IV-GICP 승리 | **{wins['IV-GICP']} / {total}** ({iv_pct:.1f}%) |",
        f"| KISS-ICP 승리 | {wins['KISS-ICP']} / {total} ({100-iv_pct:.1f}%) |",
        "",
        "### 도메인별 승패",
        "",
        "| 도메인 | 총 수 | IV-GICP 승 | KISS-ICP 승 |",
        "| --- | --- | --- | --- |",
    ]
    for d, s in domain_stats.items():
        lines.append(f"| {d} | {s['total']} | {s['iv']} | {s['kiss']} |")

    if kiss_wins:
        lines += [
            "",
            "### KISS-ICP가 이긴 케이스",
            "",
            "| Dataset | ATE IV-GICP | ATE KISS-ICP | 차이 |",
            "| --- | --- | --- | --- |",
        ]
        for label, aiv, ak, pct in kiss_wins:
            lines.append(f"| {label} | {aiv:.4f} | {ak:.4f} | +{pct:.1f}% |")

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"\nMD saved: {out_path}")


def _write_json(results, max_frames, out_path):
    import json
    data = {
        "max_frames": max_frames,
        "results": [
            {
                "label": label,
                "ate_iv": None if np.isnan(ate_iv) else round(ate_iv, 6),
                "rpe_iv": None if np.isnan(rpe_iv) else round(rpe_iv, 6),
                "ms_iv": None if np.isnan(ms_iv) else round(ms_iv, 2),
                "ate_kiss": None if np.isnan(ate_kiss) else round(ate_kiss, 6),
                "rpe_kiss": None if np.isnan(rpe_kiss) else round(rpe_kiss, 6),
                "ms_kiss": None if np.isnan(ms_kiss) else round(ms_kiss, 2),
            }
            for label, ate_iv, rpe_iv, ms_iv, ate_kiss, rpe_kiss, ms_kiss in results
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(data, indent=2))
    print(f"JSON saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="All sequences — IV-GICP vs KISS-ICP 정리")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--save-json", default=None, help="결과 JSON 저장 경로")
    parser.add_argument("--save-md", default=None, help="결과 MD 저장 경로")
    args = parser.parse_args()

    runs = build_runs()
    print(f"Total runs: {len(runs)} (each {args.max_frames} frames)\n")

    results = []
    for idx, (label, loader, loader_kw, alpha, params, kiss_params) in enumerate(runs):
        print(f"[{idx+1}/{len(runs)}] {label} ...", end=" ", flush=True)
        try:
            frames, poses_gt = loader(max_frames=args.max_frames, **loader_kw)
        except Exception as e:
            print(f"load err: {e}")
            results.append((label, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        if len(frames) < 2 or len(poses_gt) < 2:
            print("too few frames")
            results.append((label, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        try:
            ate_iv, rpe_iv, ms_iv = _run_iv_gicp(frames, poses_gt, alpha, params, args.device)
        except Exception as e:
            print(f"IV-GICP err: {e}")
            ate_iv = rpe_iv = ms_iv = np.nan
        try:
            ate_kiss, rpe_kiss, ms_kiss = _run_kiss(frames, poses_gt, **kiss_params)
        except Exception as e:
            print(f"KISS err: {e}")
            ate_kiss = rpe_kiss = ms_kiss = np.nan
        winner = "IV-GICP" if (not np.isnan(ate_iv) and (np.isnan(ate_kiss) or ate_iv < ate_kiss)) else "KISS-ICP"
        print(f"ATE IV={ate_iv:.4f} KISS={ate_kiss:.4f}  {winner}")
        results.append((label, ate_iv, rpe_iv, ms_iv, ate_kiss, rpe_kiss, ms_kiss))

    # 정리 테이블
    print("\n" + "=" * 108)
    print("  전체 시퀀스 30fr — IV-GICP vs KISS-ICP (정확도 + 속도)")
    print("=" * 108)
    print(f"  {'Dataset':<36}  {'ATE (m)':>14}  {'RPE (m)':>14}  {'속도 (ms/fr)':>16}  {'정확도':<10}")
    print(f"  {'':36}  {'IV-GICP':>6}  {'KISS':>6}  {'IV-GICP':>6}  {'KISS':>6}  {'IV-GICP':>6}  {'KISS':>6}")
    print("  " + "-" * 104)
    for label, ate_iv, rpe_iv, ms_iv, ate_kiss, rpe_kiss, ms_kiss in results:
        aiv = f"{ate_iv:.4f}" if not np.isnan(ate_iv) else "—"
        ak = f"{ate_kiss:.4f}" if not np.isnan(ate_kiss) else "—"
        riv = f"{rpe_iv:.4f}" if not np.isnan(rpe_iv) else "—"
        rk = f"{rpe_kiss:.4f}" if not np.isnan(rpe_kiss) else "—"
        miv = f"{ms_iv:.0f}" if not np.isnan(ms_iv) else "—"
        mk = f"{ms_kiss:.0f}" if not np.isnan(ms_kiss) else "—"
        if np.isnan(ate_iv):
            winner = "—"
        elif np.isnan(ate_kiss):
            winner = "IV-GICP"
        else:
            winner = "IV-GICP" if ate_iv < ate_kiss else "KISS-ICP"
        print(f"  {label:<36}  {aiv:>6}  {ak:>6}  {riv:>6}  {rk:>6}  {miv:>6}  {mk:>6}  {winner:<10}")
    print("=" * 108)

    if args.save_json:
        _write_json(results, args.max_frames, args.save_json)
    if args.save_md:
        _write_md(results, args.max_frames, args.save_md)


if __name__ == "__main__":
    main()

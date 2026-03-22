#!/usr/bin/env python3
"""
IV-GICP Full Evaluation — 전체 프레임 병렬 실행
=================================================
각 시퀀스를 끝까지 실행하고 IV-GICP vs KISS-ICP ATE/RPE를 비교합니다.
시퀀스별 독립 프로세스로 병렬 실행.

출력:
    results/full_eval/<dataset>/<seq>/results.json   — 시퀀스별
    results/full_eval/summary.json                   — 전체 요약
    results/full_eval/summary.md                     — 마크다운 표

Usage:
    uv run python examples/run_full_eval.py                        # 전체, 병렬
    uv run python examples/run_full_eval.py --workers 4            # 최대 4개 동시
    uv run python examples/run_full_eval.py --dataset kitti        # 특정 도메인
    uv run python examples/run_full_eval.py --seq "KITTI seq05"    # 단일 시퀀스
    uv run python examples/run_full_eval.py --no-parallel          # 순차 실행
    uv run python examples/run_full_eval.py --no-skip              # 기존 결과 무시 재실행
"""

import argparse
import json
import sys
import time
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

# ── Dataset paths ──────────────────────────────────────────────────────────────
KITTI_ROOT  = Path("/home/km/data/kitti/dataset")
SUBT_ROOT   = Path("/home/km/data/SubT-MRS")
GEODE_ROOT  = Path("/home/km/data/GEODE")
MULRAN_ROOT = Path("/home/km/data/MulRan")

# ── Best params per domain ─────────────────────────────────────────────────────

def _kitti_iv():
    # max_map_frames=500 keeps ~1.3km of map context for KITTI's long loop sequences.
    # mf=10 was too small → old map evicted at turns → cumulative drift.
    return dict(voxel_size=1.0, source_voxel_size=0.3, alpha=0.1,
                max_correspondence_distance=2.0, initial_threshold=2.0,
                max_map_frames=500, max_iterations=30, min_range=0.5, max_range=80.0,
                auto_alpha=False, auto_alpha_from_intensity=False,
                source_drop_small_voxels=False, source_max_output_features=0,
                source_min_feature_score=0.0, max_source_points=0)

def _kitti_k():  return dict(voxel_size=1.0,  min_range=0.5, max_range=80.0)

def _geode_urban_iv():
    # min_motion_th=0.5 (=initial_threshold/3=1.5/3): uniform concrete tunnels
    # need wider sigma floor to maintain enough correspondences (same as SubT).
    return dict(voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
                max_correspondence_distance=2.0, initial_threshold=1.5,
                max_map_frames=500, max_iterations=30, min_range=0.5, max_range=80.0,
                map_radius=80.0, min_motion_th=0.5,
                auto_alpha=False, auto_alpha_from_intensity=False,
                source_drop_small_voxels=False, source_max_output_features=0,
                source_min_feature_score=0.0, max_source_points=0)

def _geode_urban_k(): return dict(voxel_size=0.5, min_range=0.5, max_range=80.0)

def _metro_iv():
    # min_motion_th=0.5: sigma floor = initial_threshold/3 = 1.5/3 = 0.5m (KISS-ICP formula).
    # max_map_frames=200: keeps recent 200 frames; spatial eviction (60m) handles old voxels.
    return dict(voxel_size=0.5, source_voxel_size=0.2, alpha=0.5,
                max_correspondence_distance=1.5, initial_threshold=1.5,
                max_map_frames=200, max_iterations=30, min_range=0.5, max_range=60.0,
                map_radius=60.0, min_motion_th=0.5,
                auto_alpha=False, auto_alpha_from_intensity=False,
                source_drop_small_voxels=False, source_max_output_features=0,
                source_min_feature_score=0.0, max_source_points=0)

def _metro_k(): return dict(voxel_size=0.5, min_range=0.5, max_range=60.0)

def _subt_iv():
    # min_motion_th=0.5 matches KISS-ICP's initial_threshold/3 = 1.5/3 = 0.5m.
    # Without this, sigma collapses to 0.1m → adaptive_corr=0.3m → too tight
    # for sparse VLP-16 in mine junctions → poor_registration cascade failure.
    return dict(voxel_size=0.5, source_voxel_size=0.3, alpha=0.1,
                max_correspondence_distance=2.0, initial_threshold=1.5,
                max_map_frames=200, max_iterations=30, min_range=0.3, max_range=80.0,
                map_radius=200.0, min_motion_th=0.5,
                auto_alpha=False, auto_alpha_from_intensity=False,
                source_drop_small_voxels=False, source_max_output_features=0,
                source_min_feature_score=0.0, max_source_points=0)

def _subt_k(): return dict(voxel_size=0.5, min_range=0.3, max_range=80.0)

def _mulran_iv():
    # map_radius=None, max_map_frames=500: age-based eviction works better than
    # spatial for outdoor loops. Large spatial radius creates wrong far correspondences.
    return dict(voxel_size=1.0, source_voxel_size=0.3, alpha=0.1,
                max_correspondence_distance=2.0, initial_threshold=2.0,
                max_map_frames=500, max_iterations=30, min_range=0.5, max_range=80.0,
                auto_alpha=False, auto_alpha_from_intensity=False,
                source_drop_small_voxels=False, source_max_output_features=0,
                source_min_feature_score=0.0, max_source_points=0)

def _mulran_k(): return dict(voxel_size=1.0, min_range=0.5, max_range=80.0)

def _helipr_iv():
    return dict(voxel_size=1.0, source_voxel_size=0.3, alpha=0.1,
                max_correspondence_distance=2.0, initial_threshold=2.0,
                max_map_frames=500, max_iterations=30, min_range=0.5, max_range=80.0,
                map_radius=400.0,
                auto_alpha=False, auto_alpha_from_intensity=False,
                source_drop_small_voxels=False, source_max_output_features=0,
                source_min_feature_score=0.0, max_source_points=0)

def _helipr_k(): return dict(voxel_size=1.0, min_range=0.5, max_range=80.0)


# ── Sequence registry ──────────────────────────────────────────────────────────

def build_sequences():
    seqs = []

    # KITTI seq00~10
    for i in range(11):
        seq = f"{i:02d}"
        velo  = KITTI_ROOT / "sequences" / seq / "velodyne"
        poses = KITTI_ROOT / "poses" / f"{seq}.txt"
        if velo.exists() and poses.exists():
            seqs.append({"label": f"KITTI seq{seq}", "domain": "kitti",
                         "loader": "load_kitti", "loader_kw": {"seq": seq},
                         "iv_params": _kitti_iv(), "kiss_params": _kitti_k(),
                         "out_dir": f"kitti/seq{seq}"})

    # GEODE Urban_Tunnel 01/02/03
    for s in ("01", "02", "03"):
        d = GEODE_ROOT / f"sensor_data/Urban_tunnel/Urban_Tunnel{s}"
        if d.exists():
            seqs.append({"label": f"GEODE Urban_Tunnel{s}", "domain": "geode_urban",
                         "loader": "load_geode_tunnel", "loader_kw": {"seq": s},
                         "iv_params": _geode_urban_iv(), "kiss_params": _geode_urban_k(),
                         "out_dir": f"geode/Urban_Tunnel{s}"})

    # GEODE Metro Shield_tunnel 1/2/3
    for tid in (1, 2, 3):
        d = GEODE_ROOT / f"sensor_data/Metro_tunnel/Shield_tunnel{tid}_gamma"
        if d.exists():
            seqs.append({"label": f"GEODE Metro Shield_tunnel{tid}", "domain": "metro",
                         "loader": "load_metro", "loader_kw": {"tunnel_id": tid},
                         "iv_params": _metro_iv(), "kiss_params": _metro_k(),
                         "out_dir": f"geode/Metro_Shield_tunnel{tid}"})

    # MulRan
    for seq in ("DCC01", "KAIST01", "Riverside01"):
        if (MULRAN_ROOT / seq / "Ouster").exists():
            seqs.append({"label": f"MulRan {seq}", "domain": "mulran",
                         "loader": "load_mulran", "loader_kw": {"seq": seq},
                         "iv_params": _mulran_iv(), "kiss_params": _mulran_k(),
                         "out_dir": f"mulran/{seq}"})

    # SubT-MRS
    subt_keys = [
        "SubT_MRS_Final_Challenge_UGV1",  "SubT_MRS_Final_Challenge_UGV2",
        "SubT_MRS_Final_Challenge_UGV3",  "SubT_MRS_Laurel_Caverns_Handheld3",
        "SubT_MRS_Urban_Challenge_UGV1",  "SubT_MRS_Urban_Challenge_UGV2",
    ]
    for key in subt_keys:
        gt = SUBT_ROOT / "LiDAR_Inertial_Track" / key / "ground_truth_path.csv"
        if gt.exists():
            short = key[9:]
            seqs.append({"label": f"SubT {short}", "domain": "subt",
                         "loader": "load_subt", "loader_kw": {"bag_key": key},
                         "iv_params": _subt_iv(), "kiss_params": _subt_k(),
                         "out_dir": f"subt/{short}"})

    # HeLiPR
    helipr_root = Path("/home/km/data/HeLiPR")
    for seq in ("DCC05", "KAIST05", "Roundabout01"):
        if (helipr_root / seq).exists():
            seqs.append({"label": f"HeLiPR {seq}", "domain": "helipr",
                         "loader": "load_helipr", "loader_kw": {"seq": seq},
                         "iv_params": _helipr_iv(), "kiss_params": _helipr_k(),
                         "out_dir": f"helipr/{seq}"})

    return seqs


# ── Metrics ────────────────────────────────────────────────────────────────────

def ate_rmse(poses_est, poses_gt):
    n = min(len(poses_est), len(poses_gt))
    if n < 2: return float("nan")
    t_est = np.array([p[:3, 3] for p in poses_est[:n]])
    t_gt  = np.array([p[:3, 3] for p in poses_gt[:n]])
    mu_e, mu_g = t_est.mean(0), t_gt.mean(0)
    H = (t_est - mu_e).T @ (t_gt - mu_g)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3); D[2, 2] = 1.0 if np.linalg.det(Vt.T @ U.T) > 0 else -1.0
    R = Vt.T @ D @ U.T; tv = mu_g - R @ mu_e
    errs = np.array([np.linalg.norm(R @ p[:3,3] + tv - g[:3,3])
                     for p, g in zip(poses_est[:n], poses_gt[:n])])
    return float(np.sqrt(np.mean(errs**2)))


def rpe_rmse(poses_est, poses_gt, delta=1):
    n = min(len(poses_est), len(poses_gt))
    if n < delta + 1: return float("nan")
    errs = []
    for i in range(n - delta):
        dE = np.linalg.inv(poses_est[i]) @ poses_est[i + delta]
        dG = np.linalg.inv(poses_gt[i])  @ poses_gt[i + delta]
        errs.append(np.linalg.norm((np.linalg.inv(dG) @ dE)[:3, 3]))
    return float(np.sqrt(np.mean(np.array(errs)**2)))


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_kitti(seq="00", max_frames=None):
    velo_dir = KITTI_ROOT / "sequences" / seq / "velodyne"
    bins = sorted(velo_dir.glob("*.bin"))
    if max_frames: bins = bins[:max_frames]
    frames = []
    for b in bins:
        raw = np.fromfile(b, dtype=np.float32).reshape(-1, 4).astype(np.float64)
        r = np.linalg.norm(raw[:, :3], axis=1)
        frames.append(raw[(r > 0.5) & (r < 80.0)])
    gt = []
    for line in (KITTI_ROOT / "poses" / f"{seq}.txt").read_text().splitlines():
        v = list(map(float, line.split()))
        T = np.eye(4); T[:3, :] = np.array(v).reshape(3, 4)
        gt.append(T)
    n = min(len(frames), len(gt))
    return frames[:n], gt[:n]


def load_geode_tunnel(seq="01", max_frames=None):
    import run_ablation as ra
    return ra.load_geode_tunnel(max_frames=max_frames or 99999, seq=seq)


def load_metro(tunnel_id=1, max_frames=None):
    import run_ablation as ra
    return ra.load_metro(max_frames=max_frames or 99999, tunnel_id=tunnel_id)


def load_subt(bag_key="SubT_MRS_Final_Challenge_UGV1", max_frames=None):
    import run_ablation as ra
    return ra.load_subt(max_frames=max_frames or 99999, bag_key=bag_key)


def load_mulran(seq="DCC01", max_frames=None):
    seq_dir = MULRAN_ROOT / seq
    bins = sorted((seq_dir / "Ouster").glob("*.bin"))
    if max_frames: bins = bins[:max_frames]
    frames = []
    for b in bins:
        raw = np.fromfile(b, dtype=np.float32).reshape(-1, 4).astype(np.float64)
        r = np.linalg.norm(raw[:, :3], axis=1)
        frames.append(raw[(r > 0.5) & (r < 100.0)])
    gt_ts_all, gt_all = [], []
    for line in (seq_dir / "global_pose.csv").read_text().splitlines():
        v = list(map(float, line.strip().split(',')))
        R = np.array([[v[1],v[2],v[3]],[v[5],v[6],v[7]],[v[9],v[10],v[11]]])
        t = np.array([v[4],v[8],v[12]])
        T = np.eye(4); T[:3,:3]=R; T[:3,3]=t
        gt_ts_all.append(int(v[0])); gt_all.append(T)
    scan_ts = np.array([int(b.stem) for b in bins], dtype=np.int64)
    gt_ts   = np.array(gt_ts_all, dtype=np.int64)
    gt = [gt_all[int(np.argmin(np.abs(gt_ts - ts)))] for ts in scan_ts]
    return frames, gt


def load_helipr(seq="DCC05", max_frames=None):
    helipr_root = Path("/home/km/data/HeLiPR")
    seq_dir = helipr_root / seq
    for sub in ("Ouster", "os1", "lidar"):
        d = seq_dir / sub
        if d.exists():
            bins = sorted(d.glob("*.bin"))
            break
    else:
        raise FileNotFoundError(f"HeLiPR {seq}: no bin dir")
    if max_frames: bins = bins[:max_frames]
    frames = []
    for b in bins:
        raw = np.fromfile(b, dtype=np.float32).reshape(-1, 4).astype(np.float64)
        r = np.linalg.norm(raw[:, :3], axis=1)
        frames.append(raw[(r > 0.5) & (r < 100.0)])
    gt_file = seq_dir / "poses.txt"
    gt = []
    for line in gt_file.read_text().splitlines():
        v = list(map(float, line.split()))
        T = np.eye(4)
        if len(v) == 12:
            T[:3, :] = np.array(v).reshape(3, 4)
        elif len(v) == 8:
            from scipy.spatial.transform import Rotation
            T[:3, 3] = v[1:4]; T[:3, :3] = Rotation.from_quat(v[4:]).as_matrix()
        gt.append(T)
    n = min(len(frames), len(gt))
    return frames[:n], gt[:n]


LOADERS = {
    "load_kitti": load_kitti,
    "load_geode_tunnel": load_geode_tunnel,
    "load_metro": load_metro,
    "load_subt": load_subt,
    "load_mulran": load_mulran,
    "load_helipr": load_helipr,
}


# ── Worker (runs in subprocess) ────────────────────────────────────────────────

def run_sequence(cfg: dict) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    label    = cfg["label"]
    out_path = ROOT / "results" / "full_eval" / cfg["out_dir"] / "results.json"

    # Resume from cached result
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if "iv_gicp" in existing and "kiss_icp" in existing:
                print(f"  [SKIP] {label} (cached)")
                return existing
        except Exception:
            pass

    print(f"  [START] {label}")
    t_wall = time.perf_counter()

    loader_fn = LOADERS[cfg["loader"]]
    frames, poses_gt = loader_fn(**cfg["loader_kw"])
    n = len(frames)
    print(f"  [{label}] {n} frames loaded")

    # ── IV-GICP ──────────────────────────────────────────────────────────────
    from iv_gicp.pipeline import IVGICPPipeline
    iv_params = dict(cfg["iv_params"])  # copy to avoid mutating original
    iv_params.setdefault("device", "cpu")
    pipeline = IVGICPPipeline(**iv_params)
    iv_times = []
    for i, f in enumerate(frames):
        pts = np.asarray(f, dtype=np.float64)
        xyz = pts[:, :3]
        itn = pts[:, 3] if pts.shape[1] >= 4 else np.zeros(len(pts))
        t0 = time.perf_counter()
        pipeline.process_frame(xyz, itn, timestamp=float(i))
        iv_times.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 500 == 0:
            print(f"  [{label}] IV {i+1}/{n} ...", end="\r", flush=True)
    iv_poses = [p.copy() for p in pipeline.get_trajectory().poses]
    iv_ate   = ate_rmse(iv_poses, poses_gt)
    iv_rpe   = rpe_rmse(iv_poses, poses_gt)
    iv_ms    = float(np.mean(iv_times[1:])) if len(iv_times) > 1 else 0.0
    print(f"  [{label}] IV-GICP  ATE={iv_ate:.4f}m  {iv_ms:.0f}ms/fr       ")

    # ── KISS-ICP ─────────────────────────────────────────────────────────────
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    kp = cfg["kiss_params"]
    kcfg = KISSConfig()
    kcfg.data.deskew  = False
    kcfg.data.max_range  = kp.get("max_range", 80.0)
    kcfg.data.min_range  = kp.get("min_range", 0.5)
    kcfg.mapping.voxel_size = kp.get("voxel_size", 1.0)
    kiss = KissICP(config=kcfg)
    kiss_poses, kiss_times = [], []
    for i, f in enumerate(frames):
        pts = np.asarray(f[:, :3], dtype=np.float64)
        pts = pts[np.isfinite(pts).all(axis=1)]
        t0 = time.perf_counter()
        kiss.register_frame(pts, np.full(len(pts), float(i)))
        kiss_times.append((time.perf_counter() - t0) * 1000)
        kiss_poses.append(kiss.last_pose.copy())
    kiss_ate = ate_rmse(kiss_poses, poses_gt)
    kiss_rpe = rpe_rmse(kiss_poses, poses_gt)
    kiss_ms  = float(np.mean(kiss_times)) if kiss_times else 0.0
    print(f"  [{label}] KISS-ICP ATE={kiss_ate:.4f}m  {kiss_ms:.0f}ms/fr")

    winner = "IV-GICP" if iv_ate < kiss_ate else "KISS-ICP"
    pct = (kiss_ate - iv_ate) / kiss_ate * 100

    result = {
        "label": label, "domain": cfg["domain"], "n_frames": n,
        "wall_time_s": round(time.perf_counter() - t_wall, 1),
        "iv_gicp":  {"ate_m": round(iv_ate, 4), "rpe_m": round(iv_rpe, 4),
                     "ms_per_frame": round(iv_ms, 1)},
        "kiss_icp": {"ate_m": round(kiss_ate, 4), "rpe_m": round(kiss_rpe, 4),
                     "ms_per_frame": round(kiss_ms, 1)},
        "winner": winner,
        "iv_improvement_pct": round(pct, 1),
        "iv_params": cfg["iv_params"],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  [SAVED] {label}  →  {out_path.relative_to(ROOT)}")
    return result


# ── Summary ────────────────────────────────────────────────────────────────────

def write_summary(results: list, out_root: Path):
    wins = sum(1 for r in results if r.get("winner") == "IV-GICP")
    summary = {"n_sequences": len(results), "n_iv_wins": wins,
               "win_rate_pct": round(wins / len(results) * 100, 1) if results else 0,
               "sequences": results}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# IV-GICP vs KISS-ICP — Full Sequence Evaluation",
        "",
        f"**{wins}/{len(results)} sequences: IV-GICP wins ({wins/len(results)*100:.0f}%)**",
        "",
        "| Dataset | Frames | IV-ATE (m) | KISS-ATE (m) | Δ% | IV-RPE | KISS-RPE | ms/fr IV | ms/fr KISS | Winner |",
        "|---------|--------|-----------|-------------|-----|--------|---------|---------|-----------|--------|",
    ]
    for r in sorted(results, key=lambda x: x.get("domain","") + x.get("label","")):
        iv = r["iv_gicp"]; ki = r["kiss_icp"]; pct = r["iv_improvement_pct"]
        win = "**IV-GICP**" if r["winner"] == "IV-GICP" else "KISS-ICP"
        lines.append(
            f"| {r['label']} | {r['n_frames']} "
            f"| {iv['ate_m']:.4f} | {ki['ate_m']:.4f} | {pct:+.1f}% "
            f"| {iv['rpe_m']:.4f} | {ki['rpe_m']:.4f} "
            f"| {iv['ms_per_frame']:.0f} | {ki['ms_per_frame']:.0f} | {win} |"
        )
    (out_root / "summary.md").write_text("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel workers (default 6; each uses OMP_NUM_THREADS=4)")
    parser.add_argument("--dataset", default=None,
                        choices=["kitti","geode_urban","metro","subt","mulran","helipr"],
                        help="Filter by domain")
    parser.add_argument("--seq",  default=None, help="Filter by label substring")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--no-skip", action="store_true", help="Rerun even if cached")
    args = parser.parse_args()

    seqs = build_sequences()
    if args.dataset: seqs = [s for s in seqs if s["domain"] == args.dataset]
    if args.seq:     seqs = [s for s in seqs if args.seq.lower() in s["label"].lower()]
    if not seqs:
        print("No sequences found. Check paths or --dataset/--seq filter."); return

    out_root = ROOT / "results" / "full_eval"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nIV-GICP Full Evaluation")
    print(f"  Sequences : {len(seqs)}")
    print(f"  Workers   : {1 if args.no_parallel else args.workers}")
    print(f"  Output    : {out_root}\n")

    results = []

    if args.no_parallel or args.workers == 1:
        for cfg in seqs:
            results.append(run_sequence(cfg))
    else:
        # Subprocess-based parallelism: each sequence runs as independent uv process
        # This avoids C++ extension (OpenMP) conflicts in forked/spawned workers.
        def _run_subprocess(cfg):
            label = cfg["label"]
            out_path = ROOT / "results" / "full_eval" / cfg["out_dir"] / "results.json"
            if not args.no_skip and out_path.exists():
                try:
                    existing = json.loads(out_path.read_text())
                    if "iv_gicp" in existing and "kiss_icp" in existing:
                        print(f"  [SKIP] {label}")
                        return existing
                except Exception:
                    pass
            env = {**__import__("os").environ, "OMP_NUM_THREADS": "4"}
            cmd = [
                "uv", "run", "python", str(ROOT / "examples" / "run_full_eval.py"),
                "--seq", label, "--no-parallel",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
            if proc.returncode != 0:
                print(f"  ERROR {label}:\n{proc.stderr[-500:]}")
                return {"label": label, "domain": cfg["domain"], "error": proc.stderr[-200:]}
            if out_path.exists():
                return json.loads(out_path.read_text())
            return {"label": label, "domain": cfg["domain"], "error": "no output file"}

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_subprocess, cfg): cfg["label"] for cfg in seqs}
            for fut in as_completed(futs):
                label = futs[fut]
                try:
                    r = fut.result()
                    if "error" not in r:
                        results.append(r)
                        iv = r["iv_gicp"]["ate_m"]; ki = r["kiss_icp"]["ate_m"]
                        win = "IV-GICP ✓" if iv < ki else "KISS-ICP"
                        print(f"  DONE  {label:<48}  IV={iv:.4f}  KISS={ki:.4f}  {win}")
                    else:
                        print(f"  ERROR {label}: {r.get('error','')}")
                except Exception as e:
                    print(f"  ERROR {label}: {e}")

    results.sort(key=lambda x: x.get("label", ""))
    write_summary(results, out_root)

    wins = sum(1 for r in results if r.get("winner") == "IV-GICP")
    print(f"\n{'='*95}")
    print(f"  {'Dataset':<48} {'Frames':>7}  {'IV-ATE':>8}  {'KISS-ATE':>8}  {'Δ%':>7}  Winner")
    print(f"  {'-'*93}")
    for r in results:
        iv_ate = r["iv_gicp"]["ate_m"]; ki_ate = r["kiss_icp"]["ate_m"]
        pct = r["iv_improvement_pct"]
        win = "IV-GICP" if r["winner"] == "IV-GICP" else "KISS-ICP"
        print(f"  {r['label']:<48} {r['n_frames']:>7}  {iv_ate:>8.4f}  {ki_ate:>8.4f}  {pct:>+7.1f}%  {win}")
    print(f"  {'─'*93}")
    print(f"  IV-GICP wins: {wins}/{len(results)} ({wins/len(results)*100:.0f}%)")
    print(f"{'='*95}\n")
    print(f"  results/full_eval/summary.md")
    print(f"  results/full_eval/summary.json\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
IV-GICP vs KISS-ICP ms/frame 비교 (합성 데이터, 데이터셋 불필요).

사용법:
  uv run python examples/bench_iv_gicp_vs_kiss.py
  uv run python examples/bench_iv_gicp_vs_kiss.py --frames 30 --pts 6000 --device cuda
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=20, help="Number of frames")
    p.add_argument("--pts", type=int, default=4000, help="Points per frame")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    args = p.parse_args()

    np.random.seed(42)
    # 합성 프레임: 약간씩 이동하는 포인트 클라우드
    frames_4d = []
    for i in range(args.frames):
        pts = np.random.randn(args.pts, 4).astype(np.float64)
        pts[:, :3] *= 2.0
        pts[:, 3] = np.clip(np.abs(pts[:, 3]), 0.01, 1.0)
        pts[:, :3] += np.array([0.02 * i, 0.01 * i, 0.0])
        frames_4d.append(pts)
    frames_xyz = [f[:, :3].astype(np.float64) for f in frames_4d]

    # ── IV-GICP ─────────────────────────────────────────────────────────────
    from iv_gicp import IVGICPPipeline
    from iv_gicp.gpu_backend import is_gpu_available

    if args.device in ("cuda", "auto") and not is_gpu_available():
        print("Note: CUDA not available; IV-GICP will use CPU (등록은 C++ 경로).")
    pipeline = IVGICPPipeline(device=args.device)
    iv_times = []
    for i, pts in enumerate(frames_4d):
        t0 = time.perf_counter()
        pipeline.process_frame(pts, timestamp=float(i))
        iv_times.append((time.perf_counter() - t0) * 1000)
    iv_avg = float(np.mean(iv_times))
    iv_std = float(np.std(iv_times))

    # ── KISS-ICP (합성 데이터는 타임스탬프 없음 → deskew=False 필수, 아니면 C++에서 abort)
    kiss_avg = kiss_std = ratio = None
    kiss_err = None
    try:
        from kiss_icp.kiss_icp import KissICP
        from kiss_icp.config import KISSConfig

        cfg = KISSConfig()
        cfg.data.max_range = 80.0
        cfg.data.min_range = 2.0
        cfg.data.deskew = False  # 타임스탬프가 0이면 deskew=True 시 C++ preprocess에서 abort
        cfg.mapping.voxel_size = 1.0
        kiss = KissICP(config=cfg)
        kiss_times = []
        for pts in frames_xyz:
            t0 = time.perf_counter()
            kiss.register_frame(pts, np.zeros(len(pts)))
            kiss_times.append((time.perf_counter() - t0) * 1000)
        kiss_avg = float(np.mean(kiss_times))
        kiss_std = float(np.std(kiss_times))
        ratio = iv_avg / kiss_avg
    except Exception as e:
        kiss_err = str(e)

    # ── 결과 출력 ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("IV-GICP vs KISS-ICP (synthetic, no dataset)")
    print("=" * 60)
    print(f"  frames={args.frames}, pts/frame={args.pts}, device={args.device}\n")
    print(f"  IV-GICP:    {iv_avg:.1f} ± {iv_std:.1f} ms/frame")
    if kiss_avg is not None:
        print(f"  KISS-ICP:   {kiss_avg:.1f} ± {kiss_std:.1f} ms/frame")
        print(f"  Ratio:     IV-GICP / KISS-ICP = {ratio:.2f}x")
        if ratio <= 10:
            print("  → 목표(2~10배) 달성")
        elif ratio <= 15:
            print("  → 목표(2~10배) 근접")
    else:
        print(f"  KISS-ICP:   (skip: {kiss_err})")
        print("  → KITTI 데이터가 있으면: uv run python examples/run_full_eval.py --data <path>")
    print("=" * 60)


if __name__ == "__main__":
    main()

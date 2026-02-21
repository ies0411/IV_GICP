#!/usr/bin/env python3
"""
FORM vs Distribution Propagation 벤치마크

Factor Graph 최적화 후 지도 갱신:
- FORM: 점 단위 변환 p_new = T_new @ T_old^{-1} @ p  (O(n))
- Distribution: 복셀 (μ, Σ)만 Lie theory로 업데이트 (O(k), k << n)
"""

import argparse
from iv_gicp import MapRefinementBenchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=10000, help="Points per frame")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    args = parser.parse_args()

    bm = MapRefinementBenchmark(
        n_points_per_frame=args.points,
        voxel_size=args.voxel_size,
    )
    r_form, r_dist = bm.benchmark(n_frames=args.frames)
    print("=" * 50)
    print("FORM vs Distribution Propagation Benchmark")
    print("=" * 50)
    print(f"FORM:         {r_form.n_updates:>8} transforms   {r_form.time_ms:>8.2f} ms")
    print(f"Distribution: {r_dist.n_updates:>8} voxels      {r_dist.time_ms:>8.2f} ms")
    ratio = r_form.time_ms / max(r_dist.time_ms, 0.01)
    if ratio > 1:
        print(f"Distribution Propagation is {ratio:.1f}x faster")
    else:
        print(f"FORM is {1/ratio:.1f}x faster")
    print("=" * 50)


if __name__ == "__main__":
    main()

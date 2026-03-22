#!/usr/bin/env python3
"""
OpenMP 스케일링 테스트: OMP_NUM_THREADS=1,2,4,8에서 reg_ms 측정.

사용법:
  uv run python examples/bench_omp_scaling.py

각 스레드 수마다 별도 프로세스를 띄워서 측정 (OpenMP는 프로세스 시작 시 스레드 수를 읽음).
"""
import os
import sys
import subprocess

def run_one(n_threads: int, n_frames: int = 20) -> float:
    env = {**os.environ, "OMP_NUM_THREADS": str(n_threads)}
    cmd = [
        sys.executable, "-c",
        f"""
import sys
sys.path.insert(0, {repr(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))})
import numpy as np
from iv_gicp import IVGICPPipeline
np.random.seed(42)
p = IVGICPPipeline()
pts = np.random.randn(4000, 4).astype(np.float64)
pts[:, :3] *= 2.0
pts[:, 3] = np.clip(np.abs(pts[:, 3]), 0.01, 1.0)
p.process_frame(pts, timestamp=0.0)
reg = []
for i in range(1, {n_frames}):
    r = p.process_frame(pts + np.array([0.02*i, 0.01*i, 0, 0]), timestamp=float(i))
    reg.append(r.reg_ms)
print(float(__import__('numpy').mean(reg)))
"""
    ]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if out.returncode != 0:
        print(f"OMP_NUM_THREADS={n_threads} failed: {out.stderr}", file=sys.stderr)
        return -1.0
    return float(out.stdout.strip())

def main():
    n_frames = 20
    threads_list = [1, 2, 4, 8]
    print("OMP_NUM_THREADS  reg_ms_avg")
    print("-" * 25)
    for n in threads_list:
        reg_avg = run_one(n, n_frames)
        if reg_avg >= 0:
            print(f"  {n}              {reg_avg:.1f}")

if __name__ == "__main__":
    main()

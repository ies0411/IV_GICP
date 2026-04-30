#!/usr/bin/env python3
"""
Summarize full-seq benchmark results into a single table.

Reads:
  results/geode/fullseq/Urban_Tunnel{01,02,03}/summary.json  — IV + KISS
  results/geode/fullseq/genz_results.json                    — GenZ (per seq)
  results/kitti/autogate/seq00/autogate.json                 — KITTI auto-gate
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent / "results"


def pct(v, ref):
    if v is None or ref is None or ref == 0:
        return ""
    d = (v - ref) / ref * 100
    return f"{d:+.1f}%"


def main():
    # GEODE
    print("=" * 90)
    print("GEODE Urban Tunnel — FULL-SEQUENCE")
    print("=" * 90)
    geode_root = ROOT / "geode" / "fullseq"
    genz_path = geode_root / "genz_results.json"
    genz = json.loads(genz_path.read_text()) if genz_path.exists() else {"results": {}}
    genz_res = genz.get("results", {})

    header = f"{'Seq':<16} {'n_fr':>5}   {'KISS':>9}  {'IV c1_off':>9}  {'IV c1_on':>9}  {'IV g005':>9}  {'IV g020':>9}  {'IV g050':>9}  {'GenZ':>9}"
    print(header)
    print("-" * len(header))
    for seq in ["01", "02", "03"]:
        name = f"Urban_Tunnel{seq}"
        p = geode_root / name / "summary.json"
        if not p.exists():
            print(f"{name:<16} (not found)")
            continue
        r = json.loads(p.read_text())
        n = r.get("kiss", r.get("iv_c1_on", {})).get("n_frames", "?")
        kiss = r.get("kiss", {}).get("ate")
        c1_off = r.get("iv_c1_off", {}).get("ate")
        c1_on = r.get("iv_c1_on", {}).get("ate")
        g005 = r.get("iv_g005", {}).get("ate")
        g020 = r.get("iv_g020", {}).get("ate")
        g050 = r.get("iv_g050", {}).get("ate")
        gz = genz_res.get(name, {}).get("genz_icp")

        def f(v):
            return f"{v:>9.3f}" if v is not None else f"{'—':>9}"

        print(f"{name:<16} {n:>5}   {f(kiss)}  {f(c1_off)}  {f(c1_on)}  {f(g005)}  {f(g020)}  {f(g050)}  {f(gz)}")

        # vs KISS
        row_vs = f"{'  vs KISS':<16}       "
        for v in [c1_off, c1_on, g005, g020, g050, gz]:
            row_vs += f"  {pct(v, kiss):>9}"
        print(f"{row_vs}")

    # KITTI auto-gate
    print()
    print("=" * 90)
    print("KITTI seq00 — FULL-SEQUENCE  (auto-gate regression)")
    print("=" * 90)
    kp = ROOT / "kitti" / "autogate" / "seq00" / "autogate.json"
    if kp.exists():
        kr = json.loads(kp.read_text())
        print(f"{'Config':<10} {'ATE[m]':>10}  {'ms/fr':>7}  {'vs c1_off':>10}")
        ref = kr.get("c1_off", {}).get("ate")
        for k, v in kr.items():
            ate = v.get("ate")
            ms = v.get("ms_mean")
            print(f"  {k:<8} {ate:>10.4f}  {ms:>7.1f}  {pct(ate, ref):>10}")
    else:
        print("(kitti autogate.json not found)")


if __name__ == "__main__":
    main()

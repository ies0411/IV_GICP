# Ablation 100프레임 결과 정리 (C1/C2/C3)

- **정의**: C1 = FIM per-voxel 가중치, C2 = 4D intensity (α>0, auto_α), C3 = 엔트로피 기반 geo/intensity 비중
- **설정**: `run_ablation.py --dataset <kitti|subt|metro|geode> --max-frames 100 --device auto`
- **저장**: `results/ablation_results.json`

**Baseline A (GICP-Base) 설명**  
A는 **동일한 adaptive voxel map**을 사용하며, C1(FIM 가중치)·C2(intensity)·C3(entropy scale)만 끈 설정이다.  
따라서 A vs B~F는 “같은 맵 위에서 우리가 추가한 요소만 켜는” **공정한 ablation**이다.

---

## C2/C3 보강 내용 (적용 후 결과 기준)

| 항목 | 변경 |
|------|------|
| **C2** | `kappa_threshold` 15 → **10**, `kappa_scale` 5 → **7**, `alpha_floor_ratio` **0.2** (κ > threshold일 때 effective_alpha ≥ alpha_max×0.2) |
| **C3** | `entropy_scale_c` 0.3 → **0.5**, clip **[0.4, 2.5]** (기존 [0.5, 2]) |
| **SubT** | `alpha` 0.1 → **0.25** (터널 상한 상향) |

---

## 보강 후 100프레임 결과 (4개 데이터셋)

### KITTI seq00 (outdoor)

| Config | C1 | C2 | C3 | ATE(m) | RPE(m) |
|--------|----|----|-----|--------|--------|
| A: GICP-Base | - | - | - | 0.3178 | 1.2162 |
| B: +C1 | Y | - | - | **0.3119** | 1.2158 |
| C: +C2 | - | Y | - | 0.3178 | 1.2162 |
| D: C1+C2 | Y | Y | - | **0.3120** | 1.2158 |
| E: C2+C3 | - | Y | Y | 0.3178 | 1.2162 |
| F: Full | Y | Y | Y | **0.3119** | 1.2158 |

### SubT Final_UGV1 (mine/tunnel)

| Config | ATE(m) | RPE(m) |
|--------|--------|--------|
| A | 0.1783 | 0.0665 |
| B: +C1 | 0.1781 | 0.0666 |
| C: +C2 | 0.1783 | 0.0665 |
| D: C1+C2 | 0.1781 | 0.0666 |
| E: C2+C3 | 0.1783 | 0.0665 |
| F: Full | **0.1780** | 0.0666 |

### Metro (GEODE Metro Shield_tunnel1)

| Config | ATE(m) | RPE(m) |
|--------|--------|--------|
| A | **0.01158** | **0.01060** |
| B: +C1 | 0.01425 | 0.01230 |
| C: +C2 | **0.01155** | **0.01060** |
| D: C1+C2 | 0.01420 | 0.01220 |
| E: C2+C3 | **0.01155** | **0.01057** |
| F: Full | 0.01421 | 0.01225 |

### GEODE Urban_Tunnel01

| Config | ATE(m) | RPE(m) |
|--------|--------|--------|
| A | 0.1728 | 0.0806 |
| B: +C1 | **0.1667** | 0.0809 |
| C: +C2 | 0.1732 | 0.0807 |
| D: C1+C2 | **0.1674** | 0.0817 |
| E: C2+C3 | 0.1734 | 0.0807 |
| F: Full | **0.1666** | 0.0813 |

---

## 요약

| 데이터셋 | C1 개선 | C2 단독(C) | C3 포함(E) | Full(F) |
|----------|---------|------------|------------|---------|
| KITTI | ✓ B/D/F < A | C ≈ A | E ≈ A | ✓ |
| SubT | ✓ B/D 소폭 | C ≈ A | E ≈ A | ✓ F 최선 |
| Metro | C1 켜면 오히려 나쁨 | C/E ≤ A | E RPE 소폭 개선 | A/C/E가 우수 |
| GEODE | ✓ B/D/F < A | C ≈ A | E ≈ A | ✓ |

- **C1**: KITTI, SubT, GEODE에서 일관된 ATE 개선. Metro에서는 이 100fr에서 C1 켜면 악화(추가 분석 필요).
- **C2/C3**: 보강 후에도 C·E만 켰을 때 A 대비 개선은 미미. Full(F)는 C1 덕에 KITTI/SubT/GEODE에서 우위.
- **Metro**: A/C/E가 B/D/F보다 좋음 — 터널이라도 이 시퀀스·설정에서는 기하만 또는 C2/C3만이 유리할 수 있음.

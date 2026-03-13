# C1만으로도 개선을 보이게 하는 논문 전략

C1 단독이 악화되지 않고, 가능하면 baseline 대비 소폭 개선까지 보이게 하려면 아래를 조합할 수 있다.

---

## 1. (구현) 기하 정규화 — fine 복셀 공분산 퇴화 억제

**원인:** C1만 쓰면 fine 복셀의 기하 공분산이 퇴화(λ_min ≈ 0) → GICP Hessian 악화.

**대안:** AdaptiveFlatVoxelMap에서 leaf 공분산 Σ를 만들 때 **최소 고유값 하한**을 둔다.

- 고유값 분해 후 `λ_i ← max(λ_i, η·λ_max)` (예: η=0.01) 적용.
- 조건수 λ_max/λ_min ≤ 1/η 로 유계 → C1만 써도 fine 복셀이 기하만으로 안정.

**구현:** `AdaptiveFlatVoxelMap(..., min_eigenvalue_ratio=0.01)` 옵션 추가. 기본값 0 = 비활성(기존 동작), C1-only ablation 시에만 켠다.

**논문 서사:** "C1 단독 시 fine 복셀의 기하 퇴화를 막기 위해 공분산에 geometry-aware regularization(최소 고유값 하한)을 적용했다. 이렇게 하면 C1 단독도 baseline 대비 악화되지 않고, 복잡 구간 해상도 향상으로 소폭 개선이 가능하다."

---

## 2. (구현 완료) 보수적 분할 — 퇴화 fine은 coarse로 대체

**구현:** `AdaptiveFlatVoxelMap(..., max_condition_number=50.0)` 또는 파이프라인 `max_condition_number=50.0`.  
fine leaf 공분산 조건수 κ > max_condition_number 이면 해당 영역은 **coarse leaf만** 사용 (0 = 비활성).

- 효과: "적응형 해상도는 기하가 잘 잡힌 곳에만 적용" → C1만 써도 과도한 퇴화 노출이 사라짐.

**논문 서사:** "C1 단독 실험에서는 fine 복셀이 퇴화에 가까우면 coarse를 유지하는 보수적 분할을 적용했다."

---

## 3. (서사만) C1을 "C2의 enabler"로 고정

구현 변경 없이 **해석만** 조정하는 방법.

- **C1 only:** "fine 복셀의 기하 퇴화로 인해 단독 사용 시 악화된다"고 명시.
- **C1+C2:** "C1이 복잡/퇴화 구간에서 해상도를 높여 주고, C2가 그 fine 복셀에서 intensity로 퇴화를 보완한다" → **C1은 C2가 효과를 내기 위한 해상도 enabler**로 서술.

이렇게 하면 C1 단독 수치가 나쁘더라도, "설계상 C1과 C2는 상호 보완"이라는 메시지로 정리할 수 있다.

---

## 권장 조합

| 목표 | 권장 |
|------|------|
| **C1 단독도 개선/유지 수치를 논문에 넣고 싶다** | §1 **min_eigenvalue_ratio** 구현 후 ablation에 "C1 (geometry-regularized)" 한 줄 추가. |
| **구현 부담 최소화** | §3 서사로 "C1 = C2의 enabler" 강조하고, C1-only는 "C2 없이는 퇴화로 악화"로 솔직히 기술. |

---

## Adaptive voxel 추가 개선 가능 항목

| 항목 | 설명 | 비고 |
|------|------|------|
| **max_condition_number** | 퇴화 fine → coarse 대체 (위 §2) | ✅ 구현됨 |
| **min_eigenvalue_ratio** | fine 공분산 λ_min 하한 (§1) | ✅ 구현됨 |
| **adaptive_max_depth** | 3단계 이상 분할 (엔트로피 높으면 계속 나눔) | ✅ 구현됨 |
| **Adaptive τ** | 고정 τ 대신 장면별 백분위(예: 75%ile)로 τ 설정 | 미구현, 터널 등 장면에 따라 유리할 수 있음 |
| **Per-level min_points** | 더 미세 레벨에서 min_points 더 낮춤 (예: level 2에서 1) | 미구현 |
| **run_kitti CLI** | `--adaptive-max-depth`, `--entropy-threshold`, `--max-condition-number` 노출 | 선택 사항 |

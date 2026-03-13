# Alpha / Intensity 정규화 옵션

현재 `alpha`는 (1) residual 스케일 `d_I = α·(I_src - I_tgt)`, (2) photometric variance `σ_I² = α²/(grad_sq+ε)` 두 곳에 들어가고, **비용함수에서 α는 상쇄**합니다:
- Cost_int = (1/2) · d_I² · ω_I = (1/2) · α²(ΔI)² · (grad_sq+ε)/α² = (1/2)(grad_sq+ε)(ΔI)²

즉 **가중치는 (grad_sq+ε)** 이고, 문제는 **ΔI가 센서 raw 단위**(0~1, 0~10000 등)라 기하 residual(미터)과 스케일이 안 맞는다는 점입니다.

---

## 옵션 1: 데이터 기반 I_scale (권장)

- **정의**: `I_scale = robust_std(I)` 또는 `median(|I - median(I)|)` (맵 또는 첫 N프레임에서 추정).
- **Residual**: `d_I = (I_src - I_tgt) / I_scale` (무차원 또는 “1 = typical intensity diff”).
- **역할**: `alpha` 대신 **한 번만** `I_scale` 추정하면, residual이 “typical 1 unit” 단위가 되어 기하(미터)와 대략 맞춰질 수 있음.
- **구현**: `alpha = 1 / I_scale` 로 두면 기존 식과 호환 가능. 파이프라인에서 첫 프레임/맵으로 `I_scale` 추정 후 `alpha = 1 / max(I_scale, 1e-6)` 전달.

---

## 옵션 2: Information balance

- **목표**: 기하 Fisher 정보와 광도 Fisher 정보를 같은 스케일로.
- **의미**: `trace(Ω_geo)` 와 `ω_I`(typical)가 같은 order가 되도록 α(또는 σ_I²) 설정.
- **식**: typical `σ_geo²` ≈ voxel_size², typical `grad_sq` 맵에서 추정 →  
  `σ_I² ≈ σ_geo²` 로 두면  
  `α² = σ_I² · (grad_sq + ε)` → `α = sqrt(σ_geo² · (grad_sq + ε))` (typical grad_sq 사용).
- **특징**: “한 번의 기하 측정”과 “한 번의 광도 측정”이 비슷한 정보량을 갖도록 함.

---

## 옵션 3: Residual 단위 정규화 (σ로 나누기)

- **정의**: `d_I_normalized = (I_src - I_tgt) / σ_I_raw`,  
  `σ_I_raw` = 해당 voxel의 intensity 표준편차 또는 맵 전체 robust std.
- **가중치**: 이 residual에 대해 weight 1 (또는 grad_sq 기반 추가 가중).
- **효과**: residual이 “표준편차 단위”가 되어, 1σ 차이 = 1 unit으로 고정. alpha 없이 스케일을 데이터 분산으로 흡수.

---

## 권장 (코드 반영 시)

1. **단기**: `alpha` 기본값을 **데이터 기반**으로 – 첫 프레임/맵에서 `I_scale = robust_std(intensity)` 추정, `alpha = 1 / max(I_scale, ε)` (상한/하한 클리핑).
2. **중기**: 옵션 3처럼 **residual을 σ_I로 나누고** weight는 grad_sq만 쓰는 형태로 정리하면, “norm”이 명시적으로 “분산으로 정규화”로 고정됨.

이렇게 하면 alpha가 “센서/데이터에 따라 한 번만 맞춰주는 스케일”로만 쓰이거나, 아예 residual 정규화로 대체할 수 있어서 휴리스틱이 줄어듭니다.

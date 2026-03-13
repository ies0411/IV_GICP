"""
FlatVoxelMap: O(N_frame) incremental voxel map for real-time odometry.

Flat hash-grid voxel map that only processes the current frame's new points
(O(N_frame) per update), replacing per-frame full octree rebuilds.

AdaptiveFlatVoxelMap (C1 contribution):
  Extends FlatVoxelMap with information-theoretic adaptive resolution.
  Maintains coarse + fine maps simultaneously; the .leaves property returns
  fine-resolution voxels in high-entropy (complex) regions and coarse voxels
  elsewhere — directly realising the paper's Section A entropy splitting.

LocalKeyframeVoxelMap (C3 contribution):
  Stores voxel statistics in each keyframe's LOCAL sensor frame.
  The .leaves property transforms local voxels to world frame on demand:
    μ_world = R_k · μ_local + t_k,  Σ_world = R_k · Σ_local · R_k^T  (exact)
  When pose estimates improve (factor graph / smoother), calling update_poses()
  invalidates the cache so the next .leaves call gets the corrected world map —
  at O(V) cost, not O(N×W) as in FORM's raw-point reconstruction.

Interface compatibility:
  - .leaves property returns List[FlatLeaf], where FlatLeaf has
    .stats (VoxelStats) and .half_size (float) — same as OctreeNode.
  - Used by IVGICPPipeline._build_voxels_from_adaptive_map() for registration targets.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .adaptive_voxelization import VoxelStats


# ── Leaf wrapper (OctreeNode-compatible interface) ────────────────────────────


class FlatLeaf:
    """Minimal leaf mimic — exposes .stats and .half_size like OctreeNode."""

    __slots__ = ("stats", "half_size")

    def __init__(self, stats: VoxelStats, half_size: float):
        self.stats = stats
        self.half_size = half_size


# ── Per-voxel state (Welford parallel algorithm) ──────────────────────────────


@dataclass
class _VoxelState:
    n: int
    mean: np.ndarray  # (3,)
    M2: np.ndarray  # (3,3) sum of squared deviations (unnormalized)
    mean_i: float
    M2_i: float  # sum of squared intensity deviations
    last_frame: int


def _merge_voxel_state(a: _VoxelState, b: _VoxelState) -> _VoxelState:
    """Chan's parallel formula: merge two Welford states into one."""
    n = a.n + b.n
    mean = (a.n * a.mean + b.n * b.mean) / n
    delta = b.mean - a.mean
    M2 = a.M2 + b.M2 + (a.n * b.n / n) * np.outer(delta, delta)
    mean_i = (a.n * a.mean_i + b.n * b.mean_i) / n
    di = b.mean_i - a.mean_i
    M2_i = a.M2_i + b.M2_i + (a.n * b.n / n) * di * di
    return _VoxelState(
        n=n, mean=mean, M2=M2, mean_i=mean_i, M2_i=M2_i,
        last_frame=max(a.last_frame, b.last_frame),
    )


# ── FlatVoxelMap ──────────────────────────────────────────────────────────────


class FlatVoxelMap:
    """
    Incremental flat-hash voxel map with vectorized per-frame updates.

    Usage in pipeline:
        flat_map = FlatVoxelMap(voxel_size=1.0, min_points=3)
        flat_map.insert_frame(pts_world, intensities, frame_id=k)
        flat_map.evict_before(frame_id=k - max_frames)
        leaves = flat_map.leaves   # → rebuild Voxel4D targets
    """

    def __init__(
        self,
        voxel_size: float = 1.0,
        min_points: int = 3,
        max_frames: int = 30,  # keep voxels updated within last max_frames frames
    ):
        self.voxel_size = voxel_size
        self.min_points = min_points
        self.max_frames = max_frames

        self._voxels: Dict[Tuple[int, int, int], _VoxelState] = {}
        self._leaves: Optional[List[FlatLeaf]] = None  # cached, None = dirty
        self._n_frames: int = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def insert_frame(
        self,
        pts: np.ndarray,  # (N, 3) xyz in world frame
        intensities: np.ndarray,  # (N,)
        frame_id: int,
    ) -> None:
        """
        Vectorized insert of one LiDAR scan into the map. O(N log N).

        Groups points by voxel key using numpy sort + unique (no Python point loop).
        Updates each voxel's Welford statistics in O(V_new) Python iterations,
        where V_new ≪ N (typically 1/50 to 1/200 of N).
        """
        if len(pts) == 0:
            return

        vs = self.voxel_size
        # import ipdb

        # ipdb.set_trace()
        # ── 1. Vectorized voxel key assignment ───────────────────────────────
        keys_arr = np.floor(pts / vs).astype(np.int64)  # (N, 3)

        # Sort for contiguous grouping (lexsort on x, y, z)
        sort_idx = np.lexsort(keys_arr.T[::-1])
        keys_s = keys_arr[sort_idx]  # (N, 3)
        pts_s = pts[sort_idx]  # (N, 3)
        ints_s = intensities[sort_idx]  # (N,)

        # Find unique voxel groups
        _, inv_idx, counts = np.unique(
            keys_s, axis=0, return_inverse=True, return_counts=True
        )  # inv_idx is monotone (keys_s sorted), enables fast reduceat
        nv_new = len(counts)

        # Group boundaries for reduceat (contiguous, cache-friendly)
        bdry = np.concatenate([[0], np.where(np.diff(inv_idx))[0] + 1])  # (V,)

        # ── 2. Vectorized per-new-voxel statistics (reduceat, ~10x vs add.at) ─
        new_means = np.add.reduceat(pts_s, bdry) / counts[:, np.newaxis]  # (V, 3)

        centered = pts_s - new_means[inv_idx]  # (N, 3)
        outer = centered[:, :, np.newaxis] * centered[:, np.newaxis, :]  # (N,3,3)
        new_M2 = np.add.reduceat(outer.reshape(-1, 9), bdry).reshape(nv_new, 3, 3)

        new_mean_i = np.add.reduceat(ints_s, bdry) / counts  # (V,)
        ci = ints_s - new_mean_i[inv_idx]
        new_M2_i = np.add.reduceat(ci**2, bdry)  # (V,)

        # ── 3. Merge with existing voxel states (Chan parallel algorithm) ────
        first_idx = bdry  # already computed above
        # .tolist() converts entire (nv_new, 3) int64 array to Python ints at C
        # speed, then tuple() is ~3x faster than 3 separate int() calls per row.
        voxel_keys = [tuple(r) for r in keys_s[first_idx].tolist()]

        # Partition into brand-new vs. already-existing voxels
        is_new = np.array([k not in self._voxels for k in voxel_keys], dtype=bool)

        # ── 3a. New voxels: simple insert (no heavy ops) ─────────────────────
        new_vi = np.where(is_new)[0]
        for i in new_vi:
            k = voxel_keys[i]
            self._voxels[k] = _VoxelState(
                n=int(counts[i]),
                mean=new_means[i].copy(),
                M2=new_M2[i].copy(),
                mean_i=float(new_mean_i[i]),
                M2_i=float(new_M2_i[i]),
                last_frame=frame_id,
            )

        # ── 3b. Existing voxels: vectorized Chan parallel merge ───────────────
        # Instead of np.outer(delta, delta) per voxel in a Python loop,
        # batch all merges as (E, 3, 3) outer products in one numpy call.
        exist_vi = np.where(~is_new)[0]
        if len(exist_vi) > 0:
            exist_keys = [voxel_keys[i] for i in exist_vi]
            exist_s = [self._voxels[k] for k in exist_keys]

            # Gather old stats into contiguous numpy arrays (E = num existing merges)
            n_a = np.array([s.n for s in exist_s], dtype=np.float64)  # (E,)
            m_a = np.array([s.mean for s in exist_s])  # (E, 3)
            M2_a = np.array([s.M2 for s in exist_s])  # (E, 3, 3)
            mi_a = np.array([s.mean_i for s in exist_s])  # (E,)
            M2i_a = np.array([s.M2_i for s in exist_s])  # (E,)

            n_b = counts[exist_vi].astype(np.float64)  # (E,)
            m_b = new_means[exist_vi]  # (E, 3)
            M2_b = new_M2[exist_vi]  # (E, 3, 3)
            mi_b = new_mean_i[exist_vi]  # (E,)
            M2i_b = new_M2_i[exist_vi]  # (E,)

            n = n_a + n_b  # (E,)
            delta = m_b - m_a  # (E, 3)
            w = n_a * n_b / n  # (E,)

            merged_n = n
            merged_m = m_a + delta * (n_b / n)[:, np.newaxis]  # (E, 3)
            # Vectorized outer: replaces E × np.outer(delta[i], delta[i]) calls
            merged_M2 = (
                M2_a
                + M2_b
                + delta[:, :, np.newaxis] * delta[:, np.newaxis, :] * w[:, np.newaxis, np.newaxis]  # (E,3,3)
            )

            delta_i = mi_b - mi_a  # (E,)
            merged_mi = mi_a + delta_i * (n_b / n)  # (E,)
            merged_M2i = M2i_a + M2i_b + delta_i**2 * w  # (E,)

            # Write back (only Python-level attribute assignment, no heavy numpy)
            for j, s in enumerate(exist_s):
                s.n = int(merged_n[j])
                s.mean = merged_m[j]
                s.M2 = merged_M2[j]
                s.mean_i = float(merged_mi[j])
                s.M2_i = float(merged_M2i[j])
                s.last_frame = frame_id

        self._n_frames = max(self._n_frames, frame_id)
        self._leaves = None  # invalidate leaf cache

    def evict_before(self, frame_id: int) -> int:
        """
        Remove voxels not updated since frame_id (sliding window eviction).
        Returns number of evicted voxels.
        """
        old_keys = [k for k, s in self._voxels.items() if s.last_frame < frame_id]
        for k in old_keys:
            del self._voxels[k]
        if old_keys:
            self._leaves = None
        return len(old_keys)

    def apply_delta_transform(self, R: np.ndarray, t: np.ndarray) -> None:
        """
        C3: O(V) retroactive map correction after window pose smoothing.

        Applies SE(3) delta (R, t) to every voxel in-place:
            μ_new  = R @ μ_old + t
            M2_new = R @ M2_old @ R^T   (covariance rotation — Theorem 2)

        Safe for window-smoothing corrections where |ΔT| is small (sub-cm).
        Larger corrections accumulate floating-point error in M2 rotation,
        but for a K=10 window the corrections are typically <1mm so this is exact.
        """
        if not self._voxels:
            return
        means = np.array([s.mean for s in self._voxels.values()])  # (V, 3)
        M2s = np.array([s.M2 for s in self._voxels.values()])  # (V, 3, 3)
        new_means = (R @ means.T).T + t  # (V, 3)
        new_M2 = np.einsum("ij,kjl,ml->kim", R, M2s, R)  # (V, 3, 3)
        for s, nm, nM in zip(self._voxels.values(), new_means, new_M2):
            s.mean = nm
            s.M2 = nM
        self._leaves = None  # invalidate cache

    @property
    def leaves(self) -> List[FlatLeaf]:
        """
        Return list of FlatLeaf objects (compatible with OctreeNode interface).
        Cached: only rebuilt when map was modified since last access.
        """
        if self._leaves is None:
            self._leaves = self._build_leaves()
        return self._leaves

    def clear(self) -> None:
        self._voxels.clear()
        self._leaves = None
        self._n_frames = 0

    def __len__(self) -> int:
        return len(self._voxels)

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_leaves(self) -> List[FlatLeaf]:
        hs = self.voxel_size / 2.0
        _REG = 1e-6 * np.eye(3)  # pre-allocated once; shared across all leaves
        leaves = []
        for s in self._voxels.values():
            if s.n < self.min_points:
                continue
            n_safe = max(s.n - 1, 1)
            cov = s.M2 / n_safe + _REG  # creates new array; _REG never mutated
            var_i = s.M2_i / n_safe
            stats = VoxelStats(
                mean=s.mean.copy(),
                cov=cov,
                mean_intensity=s.mean_i,
                var_intensity=var_i,
                n_points=s.n,
            )
            leaves.append(FlatLeaf(stats, hs))
        return leaves


# ── AdaptiveFlatVoxelMap (C1) ─────────────────────────────────────────────────


class AdaptiveFlatVoxelMap:
    """
    C1 contribution: entropy-based adaptive resolution using multi-level flat maps.

    Maintains multiple resolution levels (base, base/2, base/4, ...) up to max_depth.
    For each voxel the combined splitting score S is computed; if S > τ_split the
    voxel is not used as a leaf and its children at the next level are considered
    instead (recursive subdivision). Thus "high entropy → keep subdividing" until
    S ≤ τ or max_depth is reached — matching the octree-style concept.

    Combined splitting criterion (paper eq. Section A):
        S = H_geo + λ · H_int  >  τ_split
    where:
        H_geo = ½ ln|(2πe)³ Σ|          (3D geometric entropy)
        H_int = ½ ln(2πe · σ_I²)         (1D intensity entropy)
        λ     = lambda_intensity (default 0.1, balances units)
        τ_split = entropy_threshold (default 2.0)

    max_depth=1: two levels (coarse + fine), same as previous two-level behaviour.
    max_depth≥2: up to 3+ levels; high-entropy regions subdivide again at each level.
    """

    def __init__(
        self,
        base_voxel_size: float = 1.0,
        fine_voxel_size: float = None,  # deprecated: ignored if max_depth set; default base/2 at level 1
        min_points_coarse: int = 3,
        min_points_fine: int = 2,
        max_frames: int = 30,
        entropy_threshold: float = 2.0,  # τ_split
        lambda_intensity: float = 0.1,  # λ weight for H_int
        # legacy alias kept for callers passing intensity_var_threshold
        intensity_var_threshold: float = None,
        # C1-only ablation: clamp min eigenvalue so fine voxels stay well-conditioned (0 = off)
        min_eigenvalue_ratio: float = 0.0,
        # number of subdivision levels (0 = coarse only, 1 = coarse+fine, 2 = coarse+mid+fine, ...)
        max_depth: int = 1,
        # C1 보수적 분할: fine leaf 공분산 조건수 κ > max_condition_number 이면 해당 영역은 coarse leaf만 사용 (0 = 비활성)
        max_condition_number: float = 0.0,
    ):
        if fine_voxel_size is None:
            fine_voxel_size = base_voxel_size / 2.0

        self.base_voxel_size = base_voxel_size
        self.fine_voxel_size = fine_voxel_size  # kept for API compat; level 1 uses base/2
        self.max_frames = max_frames
        self.entropy_threshold = entropy_threshold
        self.lambda_intensity = lambda_intensity
        self.min_eigenvalue_ratio = min_eigenvalue_ratio
        self.max_depth = max(0, int(max_depth))
        self.max_condition_number = max(0.0, float(max_condition_number))

        # Level ℓ has voxel_size = base / 2^ℓ  (ℓ = 0 .. max_depth)
        min_pts_level0 = min_points_coarse
        min_pts_rest = min_points_fine
        self._level_maps: List[FlatVoxelMap] = []
        for level in range(self.max_depth + 1):
            vs = base_voxel_size / (2.0 ** level)
            mp = min_pts_level0 if level == 0 else min_pts_rest
            self._level_maps.append(FlatVoxelMap(vs, mp, max_frames))
        self._leaves_cache = None  # None = dirty

    # ── Public interface (mirrors FlatVoxelMap) ───────────────────────────────

    def insert_frame(
        self,
        pts: np.ndarray,  # (N, 3)
        intensities: np.ndarray,  # (N,)
        frame_id: int,
    ) -> None:
        for m in self._level_maps:
            m.insert_frame(pts, intensities, frame_id)
        self._leaves_cache = None

    def evict_before(self, frame_id: int) -> None:
        for m in self._level_maps:
            m.evict_before(frame_id)
        self._leaves_cache = None

    def apply_delta_transform(self, R: np.ndarray, t: np.ndarray) -> None:
        """C3: delegate to all level maps."""
        for m in self._level_maps:
            m.apply_delta_transform(R, t)
        self._leaves_cache = None

    @property
    def leaves(self) -> List[FlatLeaf]:
        if self._leaves_cache is None:
            self._leaves_cache = self._build_adaptive_leaves()
        return self._leaves_cache

    def clear(self) -> None:
        for m in self._level_maps:
            m.clear()
        self._leaves_cache = None

    def __len__(self) -> int:
        return len(self._level_maps[0])

    # ── Private ───────────────────────────────────────────────────────────────

    _LOG_K3_HALF = 0.5 * np.log((2.0 * np.pi * np.e) ** 3)  # ½ ln|(2πe)³|
    _LOG_2PIE_HALF = 0.5 * np.log(2.0 * np.pi * np.e)  # ½ ln(2πe)

    def _combined_score(self, s: _VoxelState) -> float:
        """
        Combined splitting score S = H_geo + λ · H_int  (paper Section A).
            H_geo = ½ ln|(2πe)³ Σ|     (3D geometric entropy)
            H_int = ½ ln(2πe · σ_I²)   (1D intensity entropy)
        """
        n_safe = max(s.n - 1, 1)
        det = np.linalg.det(s.M2 / n_safe + 1e-8 * np.eye(3))
        h_geo = self._LOG_K3_HALF + 0.5 * np.log(max(det, 1e-30))
        h_int = self._LOG_2PIE_HALF + 0.5 * np.log(max(s.M2_i / n_safe, 1e-30))
        return h_geo + self.lambda_intensity * h_int

    def _regularize_cov(self, cov: np.ndarray) -> np.ndarray:
        """If min_eigenvalue_ratio > 0, clamp eigenvalues so λ_min >= η·λ_max (well-conditioned for C1-only)."""
        if self.min_eigenvalue_ratio <= 0.0:
            return cov
        eigvals, U = np.linalg.eigh(cov)
        lam_max = eigvals.max()
        lam_min_clip = self.min_eigenvalue_ratio * lam_max
        eigvals_clipped = np.maximum(eigvals, lam_min_clip)
        return U @ np.diag(eigvals_clipped) @ U.T

    def _parent_key(self, level: int, key: Tuple[int, ...]) -> Optional[Tuple[int, int, int]]:
        """Parent key at level-1 of a voxel at level (for level >= 1)."""
        if level <= 0:
            return None
        return (key[0] // 2, key[1] // 2, key[2] // 2)

    def _children_keys(self, level: int, parent_key: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Child voxel keys at level+1 that lie inside the given parent at level."""
        px, py, pz = parent_key
        out: List[Tuple[int, int, int]] = []
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    ck = (2 * px + i, 2 * py + j, 2 * pz + k)
                    if ck in self._level_maps[level + 1]._voxels:
                        out.append(ck)
        return out

    def _level0_ancestor(self, key: Tuple[int, int, int], level: int) -> Tuple[int, int, int]:
        """Level-0 voxel key that contains the given key at level."""
        if level <= 0:
            return key
        s = 2 ** level
        return (key[0] // s, key[1] // s, key[2] // s)

    def _condition_number(self, cov: np.ndarray) -> float:
        """κ = λ_max / λ_min (≥ 1). Large κ → degenerate."""
        eigvals = np.linalg.eigvalsh(cov)
        lam_min = max(eigvals.min(), 1e-12)
        lam_max = max(eigvals.max(), 1e-12)
        return float(lam_max / lam_min)

    def _build_adaptive_leaves(self) -> List[FlatLeaf]:
        _REG = 1e-6 * np.eye(3)
        leaves: List[FlatLeaf] = []

        # Level 0: all coarse voxels with n >= min_points
        map0 = self._level_maps[0]
        complex_parents: set = set()
        if self.max_depth >= 0:
            c_keys = [k for k, s in map0._voxels.items() if s.n >= map0.min_points]
            if c_keys:
                c_states = [map0._voxels[k] for k in c_keys]
                n_safe_arr = np.maximum(np.array([s.n for s in c_states]) - 1, 1).astype(np.float64)
                M2_arr = np.array([s.M2 for s in c_states])
                M2i_arr = np.array([s.M2_i for s in c_states])
                cov_arr = M2_arr / n_safe_arr[:, None, None] + 1e-8 * np.eye(3)
                dets = np.linalg.det(cov_arr)
                var_i = M2i_arr / n_safe_arr
                h_geo = self._LOG_K3_HALF + 0.5 * np.log(np.maximum(dets, 1e-30))
                h_int = self._LOG_2PIE_HALF + 0.5 * np.log(np.maximum(var_i, 1e-30))
                scores = h_geo + self.lambda_intensity * h_int
                for i in np.where(scores > self.entropy_threshold)[0]:
                    complex_parents.add(c_keys[i])
            # Simple level-0 voxels → leaves
            hs0 = (self.base_voxel_size / (2.0 ** 0)) / 2.0
            for k, s in map0._voxels.items():
                if k in complex_parents or s.n < map0.min_points:
                    continue
                n_safe = max(s.n - 1, 1)
                cov = s.M2 / n_safe + _REG
                cov = self._regularize_cov(cov)
                stats = VoxelStats(
                    mean=s.mean.copy(),
                    cov=cov,
                    mean_intensity=s.mean_i,
                    var_intensity=s.M2_i / n_safe,
                    n_points=s.n,
                )
                leaves.append(FlatLeaf(stats, hs0))

        use_coarse_level0: set = set()  # κ > max_condition_number 인 fine 영역 → coarse leaf만 사용
        hs0 = (self.base_voxel_size / (2.0 ** 0)) / 2.0

        # Levels 1 .. max_depth: only voxels whose parent was complex
        for level in range(1, self.max_depth + 1):
            m = self._level_maps[level]
            hs = (self.base_voxel_size / (2.0 ** level)) / 2.0
            if level == 1:
                candidates = set()
                for pk in complex_parents:
                    for ck in self._children_keys(0, pk):
                        if m._voxels.get(ck) and m._voxels[ck].n >= m.min_points:
                            candidates.add(ck)
            else:
                candidates = set()
                for pk in complex_parents:
                    for ck in self._children_keys(level - 1, pk):
                        if m._voxels.get(ck) and m._voxels[ck].n >= m.min_points:
                            candidates.add(ck)
            # Batch S for this level's candidates
            if not candidates:
                complex_parents = set()
                break
            c_keys = list(candidates)
            c_states = [m._voxels[k] for k in c_keys]
            n_safe_arr = np.maximum(np.array([s.n for s in c_states]) - 1, 1).astype(np.float64)
            M2_arr = np.array([s.M2 for s in c_states])
            M2i_arr = np.array([s.M2_i for s in c_states])
            cov_arr = M2_arr / n_safe_arr[:, None, None] + 1e-8 * np.eye(3)
            dets = np.linalg.det(cov_arr)
            var_i = M2i_arr / n_safe_arr
            h_geo = self._LOG_K3_HALF + 0.5 * np.log(np.maximum(dets, 1e-30))
            h_int = self._LOG_2PIE_HALF + 0.5 * np.log(np.maximum(var_i, 1e-30))
            scores = h_geo + self.lambda_intensity * h_int
            complex_parents = {c_keys[i] for i in np.where(scores > self.entropy_threshold)[0]}
            # 보수적 분할: κ > max_condition_number 이면 해당 영역은 coarse만 사용
            if self.max_condition_number > 0:
                for i in range(len(c_keys)):
                    cov = cov_arr[i]
                    if self._condition_number(cov) > self.max_condition_number:
                        use_coarse_level0.add(self._level0_ancestor(c_keys[i], level))
            # At max_depth always add as leaf; else add only simple ones, subdivide complex
            if level == self.max_depth:
                for k in candidates:
                    if self._level0_ancestor(k, level) in use_coarse_level0:
                        continue
                    s = m._voxels[k]
                    if s.n < m.min_points:
                        continue
                    n_safe = max(s.n - 1, 1)
                    cov = s.M2 / n_safe + _REG
                    cov = self._regularize_cov(cov)
                    stats = VoxelStats(
                        mean=s.mean.copy(),
                        cov=cov,
                        mean_intensity=s.mean_i,
                        var_intensity=s.M2_i / n_safe,
                        n_points=s.n,
                    )
                    leaves.append(FlatLeaf(stats, hs))
                complex_parents = set()
            else:
                for i, k in enumerate(c_keys):
                    if k in complex_parents:
                        continue
                    if self._level0_ancestor(k, level) in use_coarse_level0:
                        continue
                    s = c_states[i]
                    n_safe = max(s.n - 1, 1)
                    cov = s.M2 / n_safe + _REG
                    cov = self._regularize_cov(cov)
                    stats = VoxelStats(
                        mean=s.mean.copy(),
                        cov=cov,
                        mean_intensity=s.mean_i,
                        var_intensity=s.M2_i / n_safe,
                        n_points=s.n,
                    )
                    leaves.append(FlatLeaf(stats, hs))
                # complex_parents stays for next level's children

        # 퇴화 fine 영역 → coarse leaf로 대체 (C1 보수적 분할)
        for k in use_coarse_level0:
            if k not in map0._voxels or map0._voxels[k].n < map0.min_points:
                continue
            s = map0._voxels[k]
            n_safe = max(s.n - 1, 1)
            cov = s.M2 / n_safe + _REG
            cov = self._regularize_cov(cov)
            stats = VoxelStats(
                mean=s.mean.copy(),
                cov=cov,
                mean_intensity=s.mean_i,
                var_intensity=s.M2_i / n_safe,
                n_points=s.n,
            )
            leaves.append(FlatLeaf(stats, hs0))

        return leaves


# ── LocalKeyframeVoxelMap (C3) ────────────────────────────────────────────────


class LocalKeyframeVoxelMap:
    """
    C3 contribution: local-frame voxel storage with O(V) world map rebuild.

    Each scan's voxel statistics are accumulated in the SENSOR (local) frame.
    The world map is built on demand by applying current pose estimates:
        μ_world  = R_k · μ_local + t_k
        Σ_world  = R_k · Σ_local · R_k^T          (exact SE(3), no approximation)

    Comparison with FORM (Potokar et al., 2025):
      FORM  stores raw local keypoints → to_voxel_map() costs O(N_pts × W)
      Ours  stores voxel distributions  → _build_world_leaves() costs O(V × W)
      where V ≪ N_pts (voxelization ratio ~1:50–1:200).

    When poses improve (smoother / factor graph step), call update_poses() to
    invalidate the cache.  The next .leaves access rebuilds the world map with
    the corrected poses at O(V) cost — no raw-point reprocessing needed.

    C1+C3 combined: set adaptive=True to use AdaptiveFlatVoxelMap per keyframe,
    giving entropy-based resolution inside each local frame.
    """

    def __init__(
        self,
        voxel_size: float = 1.0,
        min_points: int = 3,
        max_frames: int = 20,
        # C1 adaptive resolution (used when adaptive=True)
        entropy_threshold: float = 2.0,  # τ_split
        lambda_intensity: float = 0.1,  # λ weight for H_int
        adaptive: bool = True,
    ):
        self.voxel_size = voxel_size
        self.min_points = min_points
        self.max_frames = max_frames
        self.entropy_threshold = entropy_threshold
        self.lambda_intensity = lambda_intensity
        self.adaptive = adaptive

        self._local_maps: Dict[int, object] = {}
        self._poses: Dict[int, np.ndarray] = {}
        self._leaves_cache: Optional[List[FlatLeaf]] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def insert_frame(
        self,
        pts_local: np.ndarray,  # (N, 3) in sensor frame
        intensities: np.ndarray,  # (N,)
        frame_id: int,
        T_world: np.ndarray,  # (4, 4) world ← sensor
    ) -> None:
        """Store local-frame voxels for this scan. O(N log N)."""
        lmap = self._make_local_map()
        lmap.insert_frame(pts_local, intensities, frame_id)
        self._local_maps[frame_id] = lmap
        self._poses[frame_id] = T_world.copy()
        self._leaves_cache = None

    def update_poses(self, poses: Dict[int, np.ndarray]) -> None:
        """
        Update pose estimates for stored keyframes (from smoother / factor graph).
        O(K). Invalidates leaf cache; next .leaves call uses corrected poses.
        Local voxel statistics are NOT touched — only the pose transforms change.
        """
        changed = False
        for fid, T in poses.items():
            if fid in self._poses:
                if not np.allclose(self._poses[fid], T, atol=1e-10):
                    self._poses[fid] = T.copy()
                    changed = True
        if changed:
            self._leaves_cache = None

    def evict_before(self, frame_id: int) -> None:
        old = [k for k in self._local_maps if k < frame_id]
        for k in old:
            del self._local_maps[k]
            del self._poses[k]
        if old:
            self._leaves_cache = None

    @property
    def leaves(self) -> List[FlatLeaf]:
        if self._leaves_cache is None:
            self._leaves_cache = self._build_world_leaves()
        return self._leaves_cache

    def clear(self) -> None:
        self._local_maps.clear()
        self._poses.clear()
        self._leaves_cache = None

    def __len__(self) -> int:
        return len(self._local_maps)

    # ── Private ───────────────────────────────────────────────────────────────

    def _make_local_map(self):
        if self.adaptive:
            return AdaptiveFlatVoxelMap(
                base_voxel_size=self.voxel_size,
                min_points_coarse=self.min_points,
                max_frames=1,
                entropy_threshold=self.entropy_threshold,
                lambda_intensity=self.lambda_intensity,
            )
        return FlatVoxelMap(self.voxel_size, self.min_points, max_frames=1)

    def _build_world_leaves(self) -> List[FlatLeaf]:
        """
        Vectorised O(V) world map construction.

        For each keyframe k with pose T_k = (R_k, t_k):
          μ_world  = R_k @ μ_local_batch + t_k          (broadcast matmul)
          Σ_world  = einsum('ij,kjl,ml->kim', R_k, Σ_batch, R_k)

        The einsum replaces k individual R@Σ@R.T calls with one BLAS kernel.
        Exact SE(3) transform — no ΔT approximation error.
        """
        _REG = 1e-6 * np.eye(3)
        leaves: List[FlatLeaf] = []

        for fid in sorted(self._local_maps):
            lmap = self._local_maps[fid]
            local_leaves = lmap.leaves
            if not local_leaves:
                continue

            T = self._poses[fid]
            R = T[:3, :3]  # (3, 3)
            t = T[:3, 3]  # (3,)

            lmeans = np.array([lf.stats.mean for lf in local_leaves])  # (k, 3)
            lcovs = np.array([lf.stats.cov for lf in local_leaves])  # (k, 3, 3)

            # Batch SE(3) transform (vectorised, no Python loop)
            wmeans = (R @ lmeans.T).T + t  # (k, 3)
            wcovs = np.einsum("ij,kjl,ml->kim", R, lcovs, R)  # (k, 3, 3)

            for j, lf in enumerate(local_leaves):
                stats = VoxelStats(
                    mean=wmeans[j],
                    cov=wcovs[j] + _REG,
                    mean_intensity=lf.stats.mean_intensity,
                    var_intensity=lf.stats.var_intensity,
                    n_points=lf.stats.n_points,
                )
                leaves.append(FlatLeaf(stats, lf.half_size))

        return leaves

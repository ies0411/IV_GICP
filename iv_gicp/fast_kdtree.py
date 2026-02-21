"""
FastKDTree: Drop-in replacement for scipy.spatial.cKDTree.

Tries to use the C++ nanoflann-based extension (iv_gicp_cpp).
Falls back to scipy.spatial.cKDTree if the extension is not compiled.

Usage:
    from iv_gicp.fast_kdtree import FastKDTree
    tree = FastKDTree(target_points)           # (N, 3) float64
    dists, indices = tree.query(queries, k=1)  # same as cKDTree.query()
"""

import numpy as np
from typing import Tuple

# Try loading the C++ extension (built with: python setup_cpp.py build_ext --inplace)
_CPP_AVAILABLE = False
try:
    from iv_gicp.cpp import iv_gicp_cpp as _cpp
    _CPP_AVAILABLE = True
except ImportError:
    pass


class FastKDTree:
    """
    KD-tree with automatic backend selection:
      - 'cpp':   nanoflann C++ (faster for large point clouds)
      - 'scipy': scipy.spatial.cKDTree (always available, fallback)

    Interface matches scipy.spatial.cKDTree for our use cases:
      tree = FastKDTree(points)
      dists, indices = tree.query(queries, k=1, distance_upper_bound=max_d)
    """

    def __init__(self, points: np.ndarray):
        """
        Build KD-tree index.

        Args:
            points: (N, 3) array of float64 points
        """
        pts = np.ascontiguousarray(points[:, :3], dtype=np.float64)

        if _CPP_AVAILABLE:
            self._tree = _cpp.KDTree3D(pts)
            self._backend = 'cpp'
        else:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(pts)
            self._backend = 'scipy'

    def query(
        self,
        points: np.ndarray,
        k: int = 1,
        distance_upper_bound: float = np.inf,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query for k nearest neighbors.

        Args:
            points: (M, 3) query points
            k:      number of neighbors
            distance_upper_bound: max search radius (returns n+1 index if no neighbor found)

        Returns:
            dists:   (M, k) or (M,) if k=1, distances
            indices: (M, k) or (M,) if k=1, indices into the tree points
        """
        q = np.ascontiguousarray(points[:, :3], dtype=np.float64)

        if self._backend == 'cpp':
            dists, indices = self._tree.query(q, k, distance_upper_bound)
            n = self._tree.size()
            # Match scipy: not-found entries → (inf, n_points)
            not_found = indices == n
            dists = dists.astype(np.float64)
            dists[not_found] = np.inf
            if k == 1:
                dists   = dists.ravel()
                indices = indices.ravel()
            return dists, indices
        else:
            return self._tree.query(q, k=k, distance_upper_bound=distance_upper_bound)

    @property
    def backend(self) -> str:
        """Returns 'cpp' or 'scipy'."""
        return self._backend

    @property
    def n(self) -> int:
        """Number of points in the tree."""
        if self._backend == 'cpp':
            return self._tree.size()
        return self._tree.n

"""SE(3) Lie algebra utilities for factor graph optimization.

Uses C++ implementation from iv_gicp_core when available (graceful fallback to Python).
"""

import numpy as np
from typing import Tuple

try:
    from .cpp import iv_gicp_core as _se3_cpp
except ImportError:
    _se3_cpp = None


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3D vector."""
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exponential map for SO(3)."""
    angle = np.linalg.norm(omega)
    if angle < 1e-8:
        return np.eye(3)
    K = skew_symmetric(omega / angle)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map for SO(3). Returns ω such that exp(ω) = R."""
    tr = np.trace(R)
    angle = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    vec = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-8:
        return vec  # first-order approx
    return angle / (2 * np.sin(angle)) * vec


def se3_exp(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SE(3) exponential map. xi = [ω, v] (6D). Returns (R, t)."""
    omega, v = xi[:3], xi[3:6]
    R = so3_exp(omega)
    angle = np.linalg.norm(omega)
    if angle < 1e-8:
        t = v.copy()
    else:
        J = (
            np.eye(3)
            + (1 - np.cos(angle)) / (angle**2) * skew_symmetric(omega)
            + (angle - np.sin(angle)) / (angle**3) * (skew_symmetric(omega) @ skew_symmetric(omega))
        )
        t = J @ v
    return R, t


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) logarithm map. Returns ξ = [ω, v] (6D)."""
    R, t = T[:3, :3], T[:3, 3]
    omega = so3_log(R)
    angle = np.linalg.norm(omega)
    if angle < 1e-8:
        return np.concatenate([omega, t])
    J_inv = (
        np.eye(3)
        - 0.5 * skew_symmetric(omega)
        + (1 - angle * np.cos(angle / 2) / (2 * np.sin(angle / 2)))
        / (angle**2)
        * (skew_symmetric(omega) @ skew_symmetric(omega))
    )
    v = J_inv @ t
    return np.concatenate([omega, v])


def se3_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 pose from R, t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def matrix_to_se3(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract R, t from 4x4 pose."""
    return T[:3, :3].copy(), T[:3, 3].copy()


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of 4x4 SE(3) pose."""
    R, t = matrix_to_se3(T)
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def se3_compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Compose two poses: T1 * T2."""
    R1, t1 = matrix_to_se3(T1)
    R2, t2 = matrix_to_se3(T2)
    return se3_to_matrix(R1 @ R2, R1 @ t2 + t1)


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Transform point p by T. p: (3,) or (N,3)."""
    p = np.asarray(p)
    if p.ndim == 1:
        return (T[:3, :3] @ p[:3]) + T[:3, 3]
    return (T[:3, :3] @ p[:, :3].T).T + T[:3, 3]


def adjoint_se3(T: np.ndarray) -> np.ndarray:
    """Adjoint of SE(3) for composing tangent vectors. 6x6."""
    R, t = matrix_to_se3(T)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = skew_symmetric(t) @ R
    return Ad


# Optional C++ backend (same interface, numpy in/out)
if _se3_cpp is not None:
    _f64 = np.float64

    def se3_exp(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R, t = _se3_cpp.se3_exp(np.asarray(xi, dtype=_f64, order="C"))
        return np.asarray(R), np.asarray(t)

    def se3_log(T: np.ndarray) -> np.ndarray:
        return np.asarray(_se3_cpp.se3_log(np.asarray(T, dtype=_f64, order="C")))

    def se3_inverse(T: np.ndarray) -> np.ndarray:
        return np.asarray(_se3_cpp.se3_inverse(np.asarray(T, dtype=_f64, order="C")))

    def se3_compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        return np.asarray(_se3_cpp.se3_compose(
            np.asarray(T1, dtype=_f64, order="C"),
            np.asarray(T2, dtype=_f64, order="C"),
        ))

    def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.asarray(_se3_cpp.transform_point(
            np.asarray(T, dtype=_f64, order="C"),
            np.asarray(p, dtype=_f64, order="C"),
        ))

    def adjoint_se3(T: np.ndarray) -> np.ndarray:
        return np.asarray(_se3_cpp.adjoint_se3(np.asarray(T, dtype=_f64, order="C")))

    def se3_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.asarray(_se3_cpp.se3_to_matrix(
            np.asarray(R, dtype=_f64, order="C"),
            np.asarray(t, dtype=_f64, order="C"),
        ))

    def matrix_to_se3(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R, t = _se3_cpp.matrix_to_se3(np.asarray(T, dtype=_f64, order="C"))
        return np.asarray(R), np.asarray(t)

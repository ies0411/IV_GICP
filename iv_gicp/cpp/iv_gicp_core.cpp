/**
 * iv_gicp_core.cpp
 *
 * C++ core for IV-GICP: full Gauss-Newton ICP loop with intensity augmentation.
 *
 * Replaces Python/GPU (torch einsum) hot-path with Eigen BLAS, eliminating:
 *   - Python GIL overhead per GN iteration
 *   - CUDA kernel launch latency (~10µs/call × 30 iters = 300µs wasted)
 *   - torch tensor allocation/indexing overhead
 *
 * Interface (pybind11, numpy in → numpy out):
 *   icp_register(src_xyz, src_int, tgt_means_4d, tgt_prec_4x4, tgt_grads,
 *                kdtree_pts, T_init, max_corr_dist, alpha, max_iter, huber_delta,
 *                min_valid)
 *   → dict{"T": (4,4), "H": (6,6), "n_valid": int, "iterations": int,
 *           "converged": bool}
 *
 * Build:
 *   python setup_cpp.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "nanoflann.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <omp.h>

namespace py = pybind11;
using Matrix4d = Eigen::Matrix4d;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;
using Matrix4x6d = Eigen::Matrix<double, 4, 6>;
using Matrix4d44 = Eigen::Matrix<double, 4, 4>;
using RowMajorMatXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


// ─── nanoflann adapter ────────────────────────────────────────────────────────

struct PointCloud {
    const double* ptr;
    size_t N;
    size_t stride;  // column stride (= 3 for (N,3) C-contiguous)

    inline size_t kdtree_get_point_count() const { return N; }
    inline double kdtree_get_pt(size_t idx, size_t dim) const {
        return ptr[idx * stride + dim];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTreeIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud>,
    PointCloud, 3>;


// ─── SE(3) helpers ────────────────────────────────────────────────────────────

// Skew-symmetric matrix [v]×
inline Matrix3d skew(const Vector3d& v) {
    Matrix3d S;
    S <<     0, -v[2],  v[1],
          v[2],     0, -v[0],
         -v[1],  v[0],     0;
    return S;
}

// SO(3) exponential map (Rodrigues)
inline Matrix3d so3_exp(const Vector3d& omega) {
    double angle = omega.norm();
    if (angle < 1e-8) return Matrix3d::Identity();
    Vector3d axis = omega / angle;
    Matrix3d K = skew(axis);
    return Matrix3d::Identity() + std::sin(angle) * K + (1.0 - std::cos(angle)) * (K * K);
}

// SE(3) exponential map: xi = [omega(3), v(3)]
inline Matrix4d se3_exp(const Vector6d& xi) {
    Vector3d omega = xi.head<3>();
    Vector3d v     = xi.tail<3>();
    double angle   = omega.norm();
    Matrix3d R     = so3_exp(omega);
    Vector3d t;
    if (angle < 1e-8) {
        t = v;
    } else {
        Matrix3d K = skew(omega / angle);
        Matrix3d J = Matrix3d::Identity()
                   + (1.0 - std::cos(angle)) / (angle * angle) * skew(omega)
                   + (angle - std::sin(angle)) / (angle * angle * angle) * (skew(omega) * skew(omega));
        t = J * v;
    }
    Matrix4d T = Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
    return T;
}


// ─── Main ICP registration ────────────────────────────────────────────────────

py::dict icp_register(
    // Source (M, 3) xyz and (M,) intensities
    py::array_t<double, py::array::c_style | py::array::forcecast> src_xyz_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_int_arr,
    // Target voxel arrays (N, 4), (N, 4, 4), (N, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> tgt_means_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> tgt_prec_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> tgt_grads_arr,
    // KDTree points (N, 3) — usually same as tgt_means_arr[:,:3]
    py::array_t<double, py::array::c_style | py::array::forcecast> kdtree_pts_arr,
    // Initial pose (4, 4)
    py::array_t<double, py::array::c_style | py::array::forcecast> T_init_arr,
    // Parameters
    double max_corr_dist,
    double alpha,
    int max_iter,
    double huber_delta,
    int min_valid
) {
    // ── Validate + map numpy buffers ─────────────────────────────────────────
    auto src_xyz_buf  = src_xyz_arr.request();
    auto src_int_buf  = src_int_arr.request();
    auto tgt_m_buf    = tgt_means_arr.request();
    auto tgt_p_buf    = tgt_prec_arr.request();
    auto tgt_g_buf    = tgt_grads_arr.request();
    auto kd_buf       = kdtree_pts_arr.request();
    auto T_buf        = T_init_arr.request();

    const int M = (int)src_xyz_buf.shape[0];   // source points
    const int N = (int)tgt_m_buf.shape[0];     // target voxels

    const double* src_xyz  = (double*)src_xyz_buf.ptr;
    const double* src_int  = (double*)src_int_buf.ptr;
    const double* tgt_m    = (double*)tgt_m_buf.ptr;    // (N, 4)
    const double* tgt_p    = (double*)tgt_p_buf.ptr;    // (N, 4, 4) = (N, 16)
    const double* tgt_g    = (double*)tgt_g_buf.ptr;    // (N, 3)
    const double* kd_pts   = (double*)kd_buf.ptr;       // (N, 3)

    // Initial pose
    Eigen::Map<const RowMajorMatXd> T_map((double*)T_buf.ptr, 4, 4);
    Matrix4d T = T_map;

    // ── Build nanoflann KDTree ───────────────────────────────────────────────
    PointCloud cloud{ kd_pts, (size_t)N, 3 };
    KDTreeIndex tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    const double max_dist_sq = max_corr_dist * max_corr_dist;

    // ── GN loop ─────────────────────────────────────────────────────────────
    Matrix6d H_last = Matrix6d::Zero();
    int n_valid_last = 0;
    int iter = 0;
    bool converged = false;

    std::vector<size_t> nn_idx(1);
    std::vector<double> nn_dist_sq(1);

    for (iter = 0; iter < max_iter; ++iter) {
        Matrix3d R = T.block<3,3>(0,0);
        Vector3d t = T.block<3,1>(0,3);

        Matrix6d H = Matrix6d::Zero();
        Vector6d b = Vector6d::Zero();
        int n_valid = 0;

        #pragma omp parallel
        {
            Matrix6d H_loc = Matrix6d::Zero();
            Vector6d b_loc = Vector6d::Zero();
            int n_loc = 0;

            // Thread-local KNN buffers (nanoflann not thread-safe across shared buffers)
            std::vector<size_t> t_idx(1);
            std::vector<double> t_dsq(1);

            #pragma omp for nowait
            for (int i = 0; i < M; ++i) {
                // Transform source point
                Vector3d p_s(src_xyz[i*3], src_xyz[i*3+1], src_xyz[i*3+2]);
                Vector3d q = R * p_s + t;                  // transformed point

                // KNN query
                nanoflann::KNNResultSet<double> rs(1);
                rs.init(t_idx.data(), t_dsq.data());
                double qd[3] = {q[0], q[1], q[2]};
                tree.findNeighbors(rs, qd, nanoflann::SearchParameters());

                if (t_dsq[0] > max_dist_sq) continue;

                int j = (int)t_idx[0];

                // Target voxel data
                Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>> Omega(tgt_p + j*16);
                Eigen::Map<const Eigen::Vector4d> mu_t(tgt_m + j*4);
                Eigen::Map<const Vector3d>        grad_t(tgt_g + j*3);

                double I_src = src_int[i];

                // 4D residual: d = [q - mu_t[:3], alpha*I_src - mu_t[3]]
                Eigen::Vector4d d;
                d.head<3>() = q - mu_t.head<3>();
                d[3]        = alpha * I_src - mu_t[3];

                // Jacobian J_xyz (3, 6): [-skew(q), I3]  for left-perturbation
                Eigen::Matrix<double, 3, 6> J_xyz;
                J_xyz.setZero();
                // Rotation part: -skew(q)
                J_xyz(0,1) =  q[2];  J_xyz(0,2) = -q[1];
                J_xyz(1,0) = -q[2];  J_xyz(1,2) =  q[0];
                J_xyz(2,0) =  q[1];  J_xyz(2,1) = -q[0];
                // Translation part: I3
                J_xyz(0,3) = 1.0; J_xyz(1,4) = 1.0; J_xyz(2,5) = 1.0;

                // Full 4D Jacobian J (4, 6)
                Matrix4x6d J;
                J.topRows<3>() = J_xyz;
                // Intensity row: J[3,:] = -alpha * grad_t^T @ J_xyz
                J.row(3) = -alpha * (grad_t.transpose() * J_xyz);

                // Huber weight
                double dist = std::sqrt(t_dsq[0]);
                double w = 1.0;
                if (huber_delta > 0.0 && dist > huber_delta) {
                    w = huber_delta / dist;
                }

                // Accumulate: H += w * J^T Omega J,  b += w * J^T Omega d
                Eigen::Matrix<double, 6, 4> JtO = w * J.transpose() * Omega;
                H_loc += JtO * J;
                b_loc += JtO * d;
                ++n_loc;
            }

            #pragma omp critical
            {
                H += H_loc;
                b += b_loc;
                n_valid += n_loc;
            }
        }

        if (n_valid < min_valid) break;

        H_last       = H;
        n_valid_last = n_valid;

        // LM-damped solve: (H + λI) dx = b
        // λ = clip(1e-4 × max|H_ii|, 1e-6, 100) — same as Python path.
        // Cap at 100 prevents over-damping on well-constrained DOFs.
        double max_diag = 0.0;
        for (int k = 0; k < 6; ++k) max_diag = std::max(max_diag, std::abs(H(k,k)));
        double lm = std::min(100.0, std::max(1e-6, 1e-4 * max_diag));
        Matrix6d Hdamp = H;
        for (int k = 0; k < 6; ++k) Hdamp(k,k) += lm;
        Vector6d dx = Hdamp.ldlt().solve(b);

        // SE(3) update: T_new = exp(-dx) @ T  (left perturbation)
        T = se3_exp(-dx) * T;

        // Convergence check
        if (dx.norm() < 1e-6) {
            converged = true;
            ++iter;
            break;
        }
    }

    // ── Pack result as numpy arrays ──────────────────────────────────────────
    auto T_out = py::array_t<double>({4, 4});
    auto H_out = py::array_t<double>({6, 6});
    {
        Eigen::Map<RowMajorMatXd> T_map2((double*)T_out.request().ptr, 4, 4);
        T_map2 = T;
        Eigen::Map<RowMajorMatXd> H_map((double*)H_out.request().ptr, 6, 6);
        H_map = H_last;
    }

    py::dict result;
    result["T"]          = T_out;
    result["H"]          = H_out;
    result["n_valid"]    = n_valid_last;
    result["iterations"] = iter;
    result["converged"]  = converged;
    return result;
}


// ─── pybind11 module ──────────────────────────────────────────────────────────

PYBIND11_MODULE(iv_gicp_core, m) {
    m.doc() = "IV-GICP C++ core: Eigen-based GN ICP loop (full registration in one call).";

    m.def("icp_register", &icp_register,
        py::arg("src_xyz"), py::arg("src_int"),
        py::arg("tgt_means"), py::arg("tgt_prec"), py::arg("tgt_grads"),
        py::arg("kdtree_pts"), py::arg("T_init"),
        py::arg("max_corr_dist") = 2.0,
        py::arg("alpha") = 0.1,
        py::arg("max_iter") = 30,
        py::arg("huber_delta") = 1.0,
        py::arg("min_valid") = 6,
        R"doc(
Full IV-GICP Gauss-Newton registration loop in C++/Eigen.

Args:
    src_xyz:       (M, 3) float64 — source points (NOT pre-transformed)
    src_int:       (M,)   float64 — source intensities [0,1]
    tgt_means:     (N, 4) float64 — target voxel [x,y,z,α·I]
    tgt_prec:      (N, 4, 4) float64 — precision matrices Ω (C⁻¹)
    tgt_grads:     (N, 3) float64 — intensity gradients ∇μ_I
    kdtree_pts:    (N, 3) float64 — target voxel centers for KDTree (= tgt_means[:,:3])
    T_init:        (4, 4) float64 — initial pose (world←sensor)
    max_corr_dist: max correspondence distance [m]
    alpha:         intensity weight (0 = geometry-only GICP)
    max_iter:      max GN iterations
    huber_delta:   Huber robust kernel threshold (0 = disabled)
    min_valid:     min correspondences to continue

Returns:
    dict with keys:
        T          (4, 4) float64 — refined pose
        H          (6, 6) float64 — Hessian from last iteration (for C3 window smoothing)
        n_valid    int — valid correspondences in last iteration
        iterations int — actual iterations run
        converged  bool — whether dx.norm() < 1e-6
)doc");
}

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
#include <memory>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <chrono>
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

// OpenMP user-defined reduction: merge (H, b, n_valid) without critical section
struct HbPair {
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    int n_valid = 0;
};
static inline void merge_Hb_combiner(HbPair& out, const HbPair& in) {
    out.H += in.H;
    out.b += in.b;
    out.n_valid += in.n_valid;
}
#pragma omp declare reduction(merge_Hb : HbPair : merge_Hb_combiner(omp_out, omp_in)) initializer(omp_priv = HbPair())

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

// SO(3) logarithm
inline Vector3d so3_log(const Matrix3d& R) {
    double tr = R.trace();
    double angle = std::acos(std::max(-1.0, std::min(1.0, (tr - 1.0) * 0.5)));
    Vector3d vec(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
    if (angle < 1e-8) return vec;
    return angle / (2.0 * std::sin(angle)) * vec;
}

// SE(3) logarithm: T -> xi (6D)
inline Vector6d se3_log(const Matrix4d& T) {
    Matrix3d R = T.block<3,3>(0,0);
    Vector3d t = T.block<3,1>(0,3);
    Vector3d omega = so3_log(R);
    double angle = omega.norm();
    Vector3d v;
    if (angle < 1e-8) {
        v = t;
    } else {
        Matrix3d J_inv = Matrix3d::Identity()
            - 0.5 * skew(omega)
            + (1.0 - angle * std::cos(angle / 2.0) / (2.0 * std::sin(angle / 2.0))) / (angle * angle)
              * (skew(omega) * skew(omega));
        v = J_inv * t;
    }
    Vector6d xi;
    xi.head<3>() = omega;
    xi.tail<3>() = v;
    return xi;
}

// SE(3) inverse, compose, transform_point, adjoint (for Python se3_utils fallback)
inline Matrix4d se3_inverse(const Matrix4d& T) {
    Matrix3d R = T.block<3,3>(0,0);
    Vector3d t = T.block<3,1>(0,3);
    Matrix4d Tinv = Matrix4d::Identity();
    Tinv.block<3,3>(0,0) = R.transpose();
    Tinv.block<3,1>(0,3) = -R.transpose() * t;
    return Tinv;
}

inline Matrix4d se3_compose(const Matrix4d& T1, const Matrix4d& T2) {
    Matrix4d out = Matrix4d::Identity();
    out.block<3,3>(0,0) = T1.block<3,3>(0,0) * T2.block<3,3>(0,0);
    out.block<3,1>(0,3) = T1.block<3,3>(0,0) * T2.block<3,1>(0,3) + T1.block<3,1>(0,3);
    return out;
}

inline Vector3d transform_point_single(const Matrix4d& T, const Vector3d& p) {
    return T.block<3,3>(0,0) * p + T.block<3,1>(0,3);
}

// Adjoint 6x6 for SE(3)
inline Eigen::Matrix<double, 6, 6> adjoint_se3(const Matrix4d& T) {
    Matrix3d R = T.block<3,3>(0,0);
    Vector3d t = T.block<3,1>(0,3);
    Eigen::Matrix<double, 6, 6> Ad;
    Ad.setZero();
    Ad.block<3,3>(0,0) = R;
    Ad.block<3,3>(3,3) = R;
    Ad.block<3,3>(3,0) = skew(t) * R;
    return Ad;
}


// ─── Voxel downsample (numpy in/out) ───────────────────────────────────────────

#include <unordered_map>
#include <algorithm>
#include <random>
#include <numeric>

py::tuple voxel_downsample(
    py::array_t<double, py::array::c_style | py::array::forcecast> points_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> intensities_arr,
    double voxel_size
) {
    py::buffer_info pb = points_arr.request();
    py::buffer_info ib = intensities_arr.request();
    if (pb.ndim != 2 || pb.shape[1] < 3) throw std::runtime_error("points must be (N, 3) or (N, 4)");
    if (ib.ndim != 1) throw std::runtime_error("intensities must be (N,)");
    size_t N = static_cast<size_t>(pb.shape[0]);
    if (static_cast<size_t>(ib.shape[0]) != N) throw std::runtime_error("points and intensities length mismatch");
    const double* pts = static_cast<const double*>(pb.ptr);
    const double* ints = static_cast<const double*>(ib.ptr);
    size_t stride = (pb.shape[1] >= 3) ? static_cast<size_t>(pb.strides[0] / sizeof(double)) : 3;

    const int64_t k0 = 73856093, k1 = 19349663, k2 = 83492791;
    struct Entry { int64_t hash; size_t idx; };
    std::vector<Entry> order(N);
    for (size_t i = 0; i < N; ++i) {
        int64_t ix = static_cast<int64_t>(std::floor(pts[i * stride + 0] / voxel_size));
        int64_t iy = static_cast<int64_t>(std::floor(pts[i * stride + 1] / voxel_size));
        int64_t iz = static_cast<int64_t>(std::floor(pts[i * stride + 2] / voxel_size));
        order[i] = { ix * k0 ^ iy * k1 ^ iz * k2, i };
    }
    std::sort(order.begin(), order.end(), [](const Entry& a, const Entry& b) { return a.hash < b.hash; });

    std::vector<double> sum_x, sum_y, sum_z, sum_i;
    std::vector<size_t> cnt;
    int64_t prev = order[0].hash;
    size_t o = order[0].idx;
    double sx = pts[o * stride + 0], sy = pts[o * stride + 1], sz = pts[o * stride + 2], si = ints[o];
    size_t nv = 1;
    for (size_t k = 1; k < N; ++k) {
        if (order[k].hash != prev) {
            sum_x.push_back(sx); sum_y.push_back(sy); sum_z.push_back(sz); sum_i.push_back(si);
            cnt.push_back(nv);
            prev = order[k].hash;
            o = order[k].idx;
            sx = pts[o * stride + 0]; sy = pts[o * stride + 1]; sz = pts[o * stride + 2]; si = ints[o];
            nv = 1;
        } else {
            o = order[k].idx;
            sx += pts[o * stride + 0]; sy += pts[o * stride + 1]; sz += pts[o * stride + 2]; si += ints[o];
            ++nv;
        }
    }
    sum_x.push_back(sx); sum_y.push_back(sy); sum_z.push_back(sz); sum_i.push_back(si);
    cnt.push_back(nv);
    size_t M = cnt.size();

    std::vector<py::ssize_t> shape_pts = { static_cast<py::ssize_t>(M), 3 };
    py::array_t<double> out_pts(shape_pts);
    py::array_t<double> out_ints(static_cast<py::ssize_t>(M));
    double* optr = out_pts.mutable_data();
    double* iptr = out_ints.mutable_data();
    for (size_t j = 0; j < M; ++j) {
        double n = static_cast<double>(cnt[j]);
        optr[j * 3 + 0] = sum_x[j] / n;
        optr[j * 3 + 1] = sum_y[j] / n;
        optr[j * 3 + 2] = sum_z[j] / n;
        iptr[j] = sum_i[j] / n;
    }
    return py::make_tuple(out_pts, out_ints);
}


// ─── GN loop (shared by icp_register and RegistrationSession) ─────────────────

struct GNResult {
    Matrix4d T;
    Matrix6d H_last;
    Vector6d v_min;           // min eigenvector of H_last (for next-frame MSCS warm-start)
    int n_valid_last = 0;
    int n_mscs_used = 0;      // correspondences used by MSCS in last iteration (0 if disabled)
    int iter = 0;
    bool converged = false;
    double gn_loop_ms = 0.0;  // profiling: total time inside GN iteration loop
};

static void run_gn_loop(
    KDTreeIndex& tree,
    int N,
    int M,
    const double* tgt_m,
    const double* tgt_p,
    const double* tgt_g,
    const double* src_xyz,
    const double* src_int,
    Matrix4d T_init,
    double max_dist_sq,
    double alpha,
    int max_iter,
    double huber_delta,
    int min_valid,
    bool use_fim_weight,
    int max_source_points,
    double gm_scale,        // Geman-McClure scale c; w=c²/(c²+r²), 0=disabled
    double fim_gate_ratio,  // hard FIM gate: remove corr. with FIM < ratio*mean, 0=disabled
    double trim_ratio,      // Trimmed GICP: drop top trim_ratio fraction by Mahal residual, 0=disabled
    bool use_mscs,          // C4: MSCS — stop when λ_min(H) ≥ λ_max/mscs_kappa_max
    double mscs_kappa_max,  // target condition number for MSCS stopping criterion
    const Vector6d& v_min_prev,  // warm-start min eigenvector from previous frame (for non-FIM path)
    GNResult& out
) {
    (void)N;
    using Mat36 = Eigen::Matrix<double, 3, 6>;
    const bool alpha_geo_only = (alpha < 1e-9);

    // Optional source subsampling for speed (0 = use all).
    // Uniform stride: deterministic, evenly-spaced indices.
    std::vector<int> source_idx;
    if (max_source_points > 0 && M > max_source_points) {
        source_idx.reserve(max_source_points);
        for (int k = 0; k < max_source_points; ++k)
            source_idx.push_back((k * M) / max_source_points);
    } else {
        source_idx.resize(M);
        std::iota(source_idx.begin(), source_idx.end(), 0);
    }
    const int M_used = (int)source_idx.size();

    auto t_gn_start = std::chrono::steady_clock::now();

    Matrix4d T = T_init;
    out.H_last = Matrix6d::Zero();
    out.n_valid_last = 0;
    out.iter = 0;
    out.converged = false;

    std::vector<Eigen::Vector4d> valid_d;
    std::vector<Mat36> valid_J_xyz;
    std::vector<Matrix4d44> valid_Omega;
    std::vector<Matrix4x6d> valid_J;
    std::vector<double> valid_w_huber;
    std::vector<Eigen::Vector3d> valid_d_3d;
    std::vector<Matrix3d> valid_Og;

    for (out.iter = 0; out.iter < max_iter; ++out.iter) {
        Matrix3d R = T.block<3,3>(0,0);
        Vector3d t = T.block<3,1>(0,3);

        Matrix6d H = Matrix6d::Zero();
        Vector6d b = Vector6d::Zero();
        int n_valid = 0;

        if (use_fim_weight) {
            valid_d.clear();
            valid_J_xyz.clear();
            valid_Omega.clear();
            valid_J.clear();
            valid_w_huber.clear();
            valid_d_3d.clear();
            valid_Og.clear();
            std::vector<double> valid_r_sq;  // per-correspondence Mahal residual (for trimming)

            // Thread-local accumulators: avoid #pragma omp critical on every correspondence.
            const int nthr = omp_get_max_threads();
            std::vector<Matrix6d> I_G_tls(static_cast<size_t>(nthr), Matrix6d::Zero());
            std::vector<std::vector<Eigen::Vector4d>> valid_d_tls(static_cast<size_t>(nthr));
            std::vector<std::vector<Mat36>> valid_J_xyz_tls(static_cast<size_t>(nthr));
            std::vector<std::vector<Matrix4d44>> valid_Omega_tls(static_cast<size_t>(nthr));
            std::vector<std::vector<Matrix4x6d>> valid_J_tls(static_cast<size_t>(nthr));
            std::vector<std::vector<double>> valid_w_huber_tls(static_cast<size_t>(nthr));
            std::vector<std::vector<double>> valid_r_sq_tls(static_cast<size_t>(nthr));
            std::vector<std::vector<Eigen::Vector3d>> valid_d_3d_tls(static_cast<size_t>(nthr));
            std::vector<std::vector<Matrix3d>> valid_Og_tls(static_cast<size_t>(nthr));

#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                std::vector<size_t> t_idx(1);
                std::vector<double> t_dsq(1);

#pragma omp for schedule(guided) nowait
                for (int si = 0; si < M_used; ++si) {
                    int i = source_idx[si];
                    if (i < 0 || i >= M) continue;
                    Vector3d p_s(src_xyz[i*3], src_xyz[i*3+1], src_xyz[i*3+2]);
                    Vector3d q = R * p_s + t;
                    nanoflann::KNNResultSet<double> rs(1);
                    rs.init(t_idx.data(), t_dsq.data());
                    double qd[3] = {q[0], q[1], q[2]};
                    tree.findNeighbors(rs, qd, nanoflann::SearchParameters());
                    if (t_dsq[0] > max_dist_sq) continue;
                    int j = (int)t_idx[0];
                    if (j < 0 || j >= N) continue;

                    Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>> Omega(tgt_p + j*16);
                    Eigen::Map<const Eigen::Vector4d> mu_t(tgt_m + j*4);
                    Matrix3d Og = Omega.block<3,3>(0,0);

                    Mat36 J_xyz;
                    J_xyz.setZero();
                    J_xyz(0,1)=q[2]; J_xyz(0,2)=-q[1]; J_xyz(1,0)=-q[2]; J_xyz(1,2)=q[0];
                    J_xyz(2,0)=q[1]; J_xyz(2,1)=-q[0];
                    J_xyz(0,3)=1.0; J_xyz(1,4)=1.0; J_xyz(2,5)=1.0;

                    double r_sq;
                    double w = 1.0;
                    if (alpha_geo_only) {
                        Eigen::Vector3d d_geo = q - mu_t.head<3>();
                        r_sq = d_geo.transpose() * Og * d_geo;
                        double r = std::sqrt(std::max(r_sq, 0.0));
                        if (gm_scale > 0.0) {
                            double c2 = gm_scale * gm_scale;
                            w = c2 / (c2 + r_sq);  // Geman-McClure: smooth outlier rejection
                        } else if (huber_delta > 0.0 && r > huber_delta) {
                            w = huber_delta / (r + 1e-9);
                        }
                        valid_d_3d_tls[tid].push_back(d_geo);
                        valid_Og_tls[tid].push_back(Og);
                    } else {
                        Eigen::Map<const Vector3d> grad_t(tgt_g + j*3);
                        Eigen::Vector4d d;
                        d.head<3>() = q - mu_t.head<3>();
                        d[3]        = alpha * src_int[i] - mu_t[3];
                        Matrix4x6d J;
                        J.topRows<3>() = J_xyz;
                        J.row(3) = -alpha * (grad_t.transpose() * J_xyz);
                        r_sq = d.transpose() * Omega * d;
                        double r = std::sqrt(std::max(r_sq, 0.0));
                        if (gm_scale > 0.0) {
                            double c2 = gm_scale * gm_scale;
                            w = c2 / (c2 + r_sq);  // Geman-McClure
                        } else if (huber_delta > 0.0 && r > huber_delta) {
                            w = huber_delta / (r + 1e-9);
                        }
                        valid_d_tls[tid].push_back(d);
                        valid_J_tls[tid].push_back(J);
                        valid_Omega_tls[tid].push_back(Omega);
                    }
                    valid_J_xyz_tls[tid].push_back(J_xyz);
                    valid_w_huber_tls[tid].push_back(w);
                    valid_r_sq_tls[tid].push_back(r_sq);
                    I_G_tls[tid] += J_xyz.transpose() * Og * J_xyz;
                }
            }

            Matrix6d I_G = Matrix6d::Zero();
            for (int t = 0; t < nthr; ++t) I_G += I_G_tls[static_cast<size_t>(t)];

            // Merge thread-local lists in deterministic thread order (preserves correspondence alignment).
            for (int t = 0; t < nthr; ++t) {
                const size_t tt = static_cast<size_t>(t);
                const size_t n = valid_J_xyz_tls[tt].size();
                for (size_t k = 0; k < n; ++k) {
                    valid_J_xyz.push_back(valid_J_xyz_tls[tt][k]);
                    valid_w_huber.push_back(valid_w_huber_tls[tt][k]);
                    valid_r_sq.push_back(valid_r_sq_tls[tt][k]);
                    if (alpha_geo_only) {
                        valid_d_3d.push_back(valid_d_3d_tls[tt][k]);
                        valid_Og.push_back(valid_Og_tls[tt][k]);
                    } else {
                        valid_d.push_back(valid_d_tls[tt][k]);
                        valid_J.push_back(valid_J_tls[tt][k]);
                        valid_Omega.push_back(valid_Omega_tls[tt][k]);
                    }
                }
            }
            n_valid = alpha_geo_only ? (int)valid_d_3d.size() : (int)valid_d.size();
            if (n_valid < min_valid) break;

            // [Trimmed GICP] Zero out w_huber for top trim_ratio fraction by Mahal residual.
            // Dynamic objects that moved between frames have large Mahal residual vs stored
            // Gaussian → trimming removes them before building H,b.
            if (trim_ratio > 0.0 && n_valid > 4) {
                std::vector<double> r_sorted = valid_r_sq;
                std::sort(r_sorted.begin(), r_sorted.end());
                size_t keep_idx = (size_t)((1.0 - trim_ratio) * r_sorted.size());
                if (keep_idx >= r_sorted.size()) keep_idx = r_sorted.size() - 1;
                double thresh = r_sorted[keep_idx];
                for (int m = 0; m < n_valid; ++m)
                    if (valid_r_sq[m] > thresh) valid_w_huber[m] = 0.0;
            }

            Eigen::SelfAdjointEigenSolver<Matrix6d> es(I_G);
            Vector6d v = es.eigenvectors().col(0);
            double sum_w_fim = 0.0;
            std::vector<double> w_fim(n_valid, 1.0);
            #pragma omp parallel for reduction(+:sum_w_fim)
            for (int m = 0; m < n_valid; ++m) {
                Matrix6d H_all;
                if (alpha_geo_only)
                    H_all = valid_J_xyz[m].transpose() * valid_Og[m] * valid_J_xyz[m];
                else
                    H_all = valid_J[m].transpose() * valid_Omega[m] * valid_J[m];
                double wm = std::max(v.dot(H_all * v), 1e-12);
                w_fim[m] = wm;
                sum_w_fim += wm;
            }
            double mean_w = sum_w_fim / (n_valid + 1e-9);
            // [I3] FIM hard gate: remove correspondences with very low FIM contribution.
            // Soft weighting (w_fim/mean_w) can bias the estimate in degenerate directions;
            // hard gating eliminates the bottom fim_gate_ratio fraction entirely.
            if (fim_gate_ratio > 0.0) {
                double gate_th = fim_gate_ratio * mean_w;
                for (int m = 0; m < n_valid; ++m)
                    if (w_fim[m] < gate_th) w_fim[m] = 0.0;
            }
            if (use_mscs && n_valid > 0) {
                // C4 MSCS (FIM path): sort by w_fim (= v^T H_m v, already computed above — no extra matrix mults).
                // Accumulate with Huber weight only (no FIM normalization) so H matches the unweighted
                // eps_target = λ_max(I_G) / κ_max. FIM weight used for ordering, not for accumulation.
                std::vector<std::pair<double,int>> scores(n_valid);
                for (int m = 0; m < n_valid; ++m)
                    scores[m] = {w_fim[m], m};  // reuse already-computed w_fim — no redundant J^T Ω J
                std::sort(scores.begin(), scores.end(), std::greater<std::pair<double,int>>());

                double lam_max_est = es.eigenvalues()[5];
                double eps_target = (mscs_kappa_max > 0.0 && lam_max_est > 0.0)
                                    ? lam_max_est / mscs_kappa_max : 0.0;

                H = Matrix6d::Zero(); b = Vector6d::Zero();
                int n_used = 0;
                for (auto& sc_m : scores) {
                    int m = sc_m.second;
                    double w = valid_w_huber[m];  // Huber only — no FIM normalization (matches unweighted eps_target)
                    if (alpha_geo_only) {
                        Eigen::Matrix<double, 6, 3> JtO = w * valid_J_xyz[m].transpose() * valid_Og[m];
                        H += JtO * valid_J_xyz[m];
                        b += JtO * valid_d_3d[m];
                    } else {
                        Eigen::Matrix<double, 6, 4> JtO = w * valid_J[m].transpose() * valid_Omega[m];
                        H += JtO * valid_J[m];
                        b += JtO * valid_d[m];
                    }
                    ++n_used;
                    if (eps_target > 0.0 && n_used % 64 == 0) {
                        Eigen::SelfAdjointEigenSolver<Matrix6d> es_chk(H);
                        if (es_chk.eigenvalues()[0] >= eps_target) break;
                    }
                }
                out.n_mscs_used = n_used;
            } else {
                HbPair Hb;
                #pragma omp parallel for reduction(merge_Hb : Hb)
                for (int m = 0; m < n_valid; ++m) {
                    double w = valid_w_huber[m] * (w_fim[m] / mean_w);
                    if (alpha_geo_only) {
                        Hb.H += w * valid_J_xyz[m].transpose() * valid_Og[m] * valid_J_xyz[m];
                        Hb.b += w * valid_J_xyz[m].transpose() * valid_Og[m] * valid_d_3d[m];
                    } else {
                        Eigen::Matrix<double, 6, 4> JtO = w * valid_J[m].transpose() * valid_Omega[m];
                        Hb.H += JtO * valid_J[m];
                        Hb.b += JtO * valid_d[m];
                    }
                }
                H = Hb.H;
                b = Hb.b;
            }
        } else if (use_mscs) {
            // C4 MSCS non-FIM path: two-pass — collect all Jacobians, then sort+greedy.
            // Uses v_min_prev (previous frame's min eigenvector) as warm-start direction.
            using Mat36 = Eigen::Matrix<double, 3, 6>;
            std::vector<Mat36> nf_J_xyz;
            std::vector<Matrix4x6d> nf_J;
            std::vector<Eigen::Vector3d> nf_d_3d;
            std::vector<Eigen::Vector4d> nf_d;
            std::vector<Matrix3d> nf_Og;
            std::vector<Matrix4d44> nf_Omega;
            std::vector<double> nf_w;

            // Single-threaded collection (thread-safe; MSCS overhead is small vs. parallel KDTree)
            std::vector<size_t> t_idx_nf(1);
            std::vector<double> t_dsq_nf(1);
            for (int si = 0; si < M_used; ++si) {
                int i = source_idx[si];
                if (i < 0 || i >= M) continue;
                Vector3d p_s(src_xyz[i*3], src_xyz[i*3+1], src_xyz[i*3+2]);
                Vector3d q = R * p_s + t;
                nanoflann::KNNResultSet<double> rs(1);
                rs.init(t_idx_nf.data(), t_dsq_nf.data());
                double qd[3] = {q[0], q[1], q[2]};
                tree.findNeighbors(rs, qd, nanoflann::SearchParameters());
                if (t_dsq_nf[0] > max_dist_sq) continue;
                int j = (int)t_idx_nf[0];
                if (j < 0 || j >= N) continue;

                Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>> Omega(tgt_p + j*16);
                Eigen::Map<const Eigen::Vector4d> mu_t(tgt_m + j*4);
                Matrix3d Og = Omega.block<3,3>(0,0);

                Mat36 J_xyz;
                J_xyz.setZero();
                J_xyz(0,1)=q[2]; J_xyz(0,2)=-q[1]; J_xyz(1,0)=-q[2]; J_xyz(1,2)=q[0];
                J_xyz(2,0)=q[1]; J_xyz(2,1)=-q[0];
                J_xyz(0,3)=1.0; J_xyz(1,4)=1.0; J_xyz(2,5)=1.0;

                double r_sq;
                double w = 1.0;
                if (alpha_geo_only) {
                    Eigen::Vector3d d_geo = q - mu_t.head<3>();
                    r_sq = d_geo.transpose() * Og * d_geo;
                    double r = std::sqrt(std::max(r_sq, 0.0));
                    if (gm_scale > 0.0) {
                        double c2 = gm_scale * gm_scale; w = c2 / (c2 + r_sq);
                    } else if (huber_delta > 0.0 && r > huber_delta) {
                        w = huber_delta / (r + 1e-9);
                    }
                    nf_d_3d.push_back(d_geo);
                    nf_Og.push_back(Og);
                } else {
                    Eigen::Map<const Vector3d> grad_t(tgt_g + j*3);
                    Eigen::Vector4d d;
                    d.head<3>() = q - mu_t.head<3>();
                    d[3]        = alpha * src_int[i] - mu_t[3];
                    Matrix4x6d J;
                    J.topRows<3>() = J_xyz;
                    J.row(3) = -alpha * (grad_t.transpose() * J_xyz);
                    r_sq = d.transpose() * Omega * d;
                    double r = std::sqrt(std::max(r_sq, 0.0));
                    if (gm_scale > 0.0) {
                        double c2 = gm_scale * gm_scale; w = c2 / (c2 + r_sq);
                    } else if (huber_delta > 0.0 && r > huber_delta) {
                        w = huber_delta / (r + 1e-9);
                    }
                    nf_d.push_back(d);
                    nf_J.push_back(J);
                    nf_Omega.push_back(Omega);
                }
                nf_J_xyz.push_back(J_xyz);
                nf_w.push_back(w);
            }
            n_valid = alpha_geo_only ? (int)nf_d_3d.size() : (int)nf_d.size();

            if (n_valid > 0) {
                // Score by s_m = v_min_prev^T · H_m · v_min_prev (warm-start from prev frame)
                std::vector<std::pair<double,int>> scores(n_valid);
                for (int m = 0; m < n_valid; ++m) {
                    Matrix6d H_m;
                    if (alpha_geo_only)
                        H_m = nf_J_xyz[m].transpose() * nf_Og[m] * nf_J_xyz[m];
                    else
                        H_m = nf_J[m].transpose() * nf_Omega[m] * nf_J[m];
                    scores[m] = {v_min_prev.dot(H_m * v_min_prev), m};
                }
                std::sort(scores.begin(), scores.end(), std::greater<std::pair<double,int>>());

                // Estimate λ_max from full I_G for ε_target (one pass, cheap)
                Matrix6d I_G_nf = Matrix6d::Zero();
                for (int m = 0; m < n_valid; ++m) {
                    if (alpha_geo_only)
                        I_G_nf += nf_J_xyz[m].transpose() * nf_Og[m] * nf_J_xyz[m];
                    else
                        I_G_nf += nf_J[m].transpose() * nf_Omega[m] * nf_J[m];
                }
                Eigen::SelfAdjointEigenSolver<Matrix6d> es_nf(I_G_nf);
                double lam_max_nf = es_nf.eigenvalues()[5];
                double eps_target_nf = (mscs_kappa_max > 0.0 && lam_max_nf > 0.0)
                                       ? lam_max_nf / mscs_kappa_max : 0.0;

                H = Matrix6d::Zero(); b = Vector6d::Zero();
                int n_used = 0;
                for (auto& sc_m : scores) {
                    int m = sc_m.second;
                    double w = nf_w[m];
                    if (alpha_geo_only) {
                        Eigen::Matrix<double, 6, 3> JtO = w * nf_J_xyz[m].transpose() * nf_Og[m];
                        H += JtO * nf_J_xyz[m];
                        b += JtO * nf_d_3d[m];
                    } else {
                        Eigen::Matrix<double, 6, 4> JtO = w * nf_J[m].transpose() * nf_Omega[m];
                        H += JtO * nf_J[m];
                        b += JtO * nf_d[m];
                    }
                    ++n_used;
                    if (eps_target_nf > 0.0 && n_used % 64 == 0) {
                        Eigen::SelfAdjointEigenSolver<Matrix6d> es_chk(H);
                        if (es_chk.eigenvalues()[0] >= eps_target_nf) break;
                    }
                }
                out.n_mscs_used = n_used;
            }
        } else {
            HbPair Hb;
            std::vector<size_t> t_idx(1);
            std::vector<double> t_dsq(1);
            #pragma omp parallel for reduction(merge_Hb : Hb) schedule(guided) firstprivate(t_idx, t_dsq)
            for (int si = 0; si < M_used; ++si) {
                int i = source_idx[si];
                if (i < 0 || i >= M) continue;
                Vector3d p_s(src_xyz[i*3], src_xyz[i*3+1], src_xyz[i*3+2]);
                Vector3d q = R * p_s + t;
                nanoflann::KNNResultSet<double> rs(1);
                rs.init(t_idx.data(), t_dsq.data());
                double qd[3] = {q[0], q[1], q[2]};
                tree.findNeighbors(rs, qd, nanoflann::SearchParameters());
                if (t_dsq[0] > max_dist_sq) continue;
                int j = (int)t_idx[0];
                if (j < 0 || j >= N) continue;

                Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>> Omega(tgt_p + j*16);
                Eigen::Map<const Eigen::Vector4d> mu_t(tgt_m + j*4);
                Matrix3d Og = Omega.block<3,3>(0,0);

                Eigen::Matrix<double, 3, 6> J_xyz;
                J_xyz.setZero();
                J_xyz(0,1)=q[2]; J_xyz(0,2)=-q[1]; J_xyz(1,0)=-q[2]; J_xyz(1,2)=q[0];
                J_xyz(2,0)=q[1]; J_xyz(2,1)=-q[0];
                J_xyz(0,3)=1.0; J_xyz(1,4)=1.0; J_xyz(2,5)=1.0;

                double r_sq;
                double w = 1.0;
                if (alpha_geo_only) {
                    Eigen::Vector3d d_geo = q - mu_t.head<3>();
                    r_sq = d_geo.transpose() * Og * d_geo;
                    double r = std::sqrt(std::max(r_sq, 0.0));
                    if (gm_scale > 0.0) {
                        double c2 = gm_scale * gm_scale;
                        w = c2 / (c2 + r_sq);
                    } else if (huber_delta > 0.0 && r > huber_delta) {
                        w = huber_delta / (r + 1e-9);
                    }
                    Eigen::Matrix<double, 6, 3> JtO = w * J_xyz.transpose() * Og;
                    Hb.H += JtO * J_xyz;
                    Hb.b += JtO * d_geo;
                } else {
                    Eigen::Map<const Vector3d> grad_t(tgt_g + j*3);
                    Eigen::Vector4d d;
                    d.head<3>() = q - mu_t.head<3>();
                    d[3]        = alpha * src_int[i] - mu_t[3];
                    Matrix4x6d J;
                    J.topRows<3>() = J_xyz;
                    J.row(3) = -alpha * (grad_t.transpose() * J_xyz);
                    r_sq = d.transpose() * Omega * d;
                    double r = std::sqrt(std::max(r_sq, 0.0));
                    if (gm_scale > 0.0) {
                        double c2 = gm_scale * gm_scale;
                        w = c2 / (c2 + r_sq);
                    } else if (huber_delta > 0.0 && r > huber_delta) {
                        w = huber_delta / (r + 1e-9);
                    }
                    Eigen::Matrix<double, 6, 4> JtO = w * J.transpose() * Omega;
                    Hb.H += JtO * J;
                    Hb.b += JtO * d;
                }
                Hb.n_valid++;
            }
            H = Hb.H;
            b = Hb.b;
            n_valid = Hb.n_valid;
        }

        if (n_valid < min_valid) break;

        out.H_last = H;
        out.n_valid_last = n_valid;

        double max_diag = 0.0;
        for (int k = 0; k < 6; ++k) max_diag = std::max(max_diag, std::abs(H(k,k)));
        double lm = std::min(100.0, std::max(1e-6, 1e-4 * max_diag));
        Matrix6d Hdamp = H;
        for (int k = 0; k < 6; ++k) Hdamp(k,k) += lm;
        Vector6d dx = Hdamp.ldlt().solve(b);

        T = se3_exp(-dx) * T;

        if (dx.norm() < 1e-6) {
            out.converged = true;
            ++out.iter;
            break;
        }
    }

    out.gn_loop_ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gn_start).count();
    out.T = T;

    // Compute min eigenvector of final Hessian for next-frame MSCS warm-start.
    {
        Matrix6d Hs = (out.H_last + out.H_last.transpose()) * 0.5;
        Eigen::SelfAdjointEigenSolver<Matrix6d> es_final(Hs);
        out.v_min = es_final.eigenvectors().col(0);
    }
}

// ─── RegistrationSession: cache target KDTree for reuse across frames ───────────

class RegistrationSession {
public:
    explicit RegistrationSession(py::array_t<double, py::array::c_style | py::array::forcecast> means_3d_arr) {
        auto buf = means_3d_arr.request();
        if (buf.ndim != 2 || buf.shape[1] != 3)
            throw std::runtime_error("RegistrationSession: means_3d must be (N, 3)");
        N_ = (int)buf.shape[0];
        const double* ptr = (const double*)buf.ptr;
        m3_.assign(ptr, ptr + N_ * 3);
        cloud_ = PointCloud{m3_.data(), (size_t)N_, 3};
        tree_ = std::make_unique<KDTreeIndex>(3, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree_->buildIndex();
    }

    py::dict register_(
        py::array_t<double, py::array::c_style | py::array::forcecast> src_xyz_arr,
        py::array_t<double, py::array::c_style | py::array::forcecast> src_int_arr,
        py::array_t<double, py::array::c_style | py::array::forcecast> tgt_means_arr,
        py::array_t<double, py::array::c_style | py::array::forcecast> tgt_prec_arr,
        py::array_t<double, py::array::c_style | py::array::forcecast> tgt_grads_arr,
        py::array_t<double, py::array::c_style | py::array::forcecast> T_init_arr,
        double max_corr_dist,
        double alpha,
        int max_iter,
        double huber_delta,
        int min_valid,
        bool use_fim_weight,
        int max_source_points = 0,
        double gm_scale = 0.0,
        double fim_gate_ratio = 0.0,
        double trim_ratio = 0.0,
        bool use_mscs = false,
        double mscs_kappa_max = 100.0,
        py::array_t<double, py::array::c_style | py::array::forcecast> v_min_prev_arr = py::array_t<double>(6)
    ) {
        auto src_xyz_buf  = src_xyz_arr.request();
        auto src_int_buf  = src_int_arr.request();
        auto tgt_m_buf    = tgt_means_arr.request();
        auto tgt_p_buf    = tgt_prec_arr.request();
        auto tgt_g_buf    = tgt_grads_arr.request();
        auto T_buf        = T_init_arr.request();

        const int M = (int)src_xyz_buf.shape[0];
        const double* src_xyz = (const double*)src_xyz_buf.ptr;
        const double* src_int = (const double*)src_int_buf.ptr;
        const double* tgt_m   = (const double*)tgt_m_buf.ptr;
        const double* tgt_p   = (const double*)tgt_p_buf.ptr;
        const double* tgt_g   = (const double*)tgt_g_buf.ptr;

        Eigen::Map<const RowMajorMatXd> T_map((const double*)T_buf.ptr, 4, 4);
        Matrix4d T_init = T_map;

        const double max_dist_sq = max_corr_dist * max_corr_dist;

        Vector6d v_prev = Vector6d::Zero();
        auto vbuf = v_min_prev_arr.request();
        if (vbuf.size == 6) std::memcpy(v_prev.data(), vbuf.ptr, 6 * sizeof(double));

        GNResult gr;
        run_gn_loop(*tree_, N_, M, tgt_m, tgt_p, tgt_g, src_xyz, src_int,
                    T_init, max_dist_sq, alpha, max_iter, huber_delta, min_valid, use_fim_weight,
                    max_source_points, gm_scale, fim_gate_ratio, trim_ratio,
                    use_mscs, mscs_kappa_max, v_prev, gr);

        auto T_out = py::array_t<double>({4, 4});
        auto H_out = py::array_t<double>({6, 6});
        auto v_min_out = py::array_t<double>(6);
        {
            Eigen::Map<RowMajorMatXd> T_map2((double*)T_out.request().ptr, 4, 4);
            T_map2 = gr.T;
            Eigen::Map<RowMajorMatXd> H_map((double*)H_out.request().ptr, 6, 6);
            H_map = gr.H_last;
            std::memcpy(v_min_out.mutable_data(), gr.v_min.data(), 6 * sizeof(double));
        }

        py::dict result;
        result["T"]             = T_out;
        result["H"]             = H_out;
        result["v_min"]         = v_min_out;
        result["n_valid"]       = gr.n_valid_last;
        result["n_mscs_used"]   = gr.n_mscs_used;
        result["iterations"]    = gr.iter;
        result["converged"]     = gr.converged;
        result["tree_build_ms"] = 0.0;  // reused session, no tree build
        result["gn_loop_ms"]    = gr.gn_loop_ms;
        return result;
    }

private:
    std::vector<double> m3_;
    PointCloud cloud_;
    std::unique_ptr<KDTreeIndex> tree_;
    int N_;
};

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
    int min_valid,
    bool use_fim_weight = false,
    int max_source_points = 0,
    double gm_scale = 0.0,
    double fim_gate_ratio = 0.0,
    double trim_ratio = 0.0,
    bool use_mscs = false,
    double mscs_kappa_max = 100.0,
    py::array_t<double, py::array::c_style | py::array::forcecast> v_min_prev_arr = py::array_t<double>(6)
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
    auto t_tree_start = std::chrono::steady_clock::now();
    PointCloud cloud{ kd_pts, (size_t)N, 3 };
    KDTreeIndex tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();
    double tree_build_ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t_tree_start).count();

    const double max_dist_sq = max_corr_dist * max_corr_dist;

    Vector6d v_prev = Vector6d::Zero();
    {
        auto vbuf = v_min_prev_arr.request();
        if (vbuf.size == 6) std::memcpy(v_prev.data(), vbuf.ptr, 6 * sizeof(double));
    }

    GNResult gr;
    run_gn_loop(tree, N, M, tgt_m, tgt_p, tgt_g, src_xyz, src_int,
                T, max_dist_sq, alpha, max_iter, huber_delta, min_valid, use_fim_weight,
                max_source_points, gm_scale, fim_gate_ratio, trim_ratio,
                use_mscs, mscs_kappa_max, v_prev, gr);

    // ── Pack result as numpy arrays ──────────────────────────────────────────
    auto T_out = py::array_t<double>({4, 4});
    auto H_out = py::array_t<double>({6, 6});
    auto v_min_out = py::array_t<double>(6);
    {
        Eigen::Map<RowMajorMatXd> T_map2((double*)T_out.request().ptr, 4, 4);
        T_map2 = gr.T;
        Eigen::Map<RowMajorMatXd> H_map((double*)H_out.request().ptr, 6, 6);
        H_map = gr.H_last;
        std::memcpy(v_min_out.mutable_data(), gr.v_min.data(), 6 * sizeof(double));
    }

    py::dict result;
    result["T"]             = T_out;
    result["H"]             = H_out;
    result["v_min"]         = v_min_out;
    result["n_valid"]       = gr.n_valid_last;
    result["n_mscs_used"]   = gr.n_mscs_used;
    result["iterations"]    = gr.iter;
    result["converged"]     = gr.converged;
    result["tree_build_ms"] = tree_build_ms;
    result["gn_loop_ms"]    = gr.gn_loop_ms;
    return result;
}


// ─── SE3 / voxel Python wrappers (numpy in/out) ─────────────────────────────────

py::tuple py_se3_exp(py::array_t<double, py::array::c_style | py::array::forcecast> xi_arr) {
    if (xi_arr.size() != 6) throw std::runtime_error("se3_exp: xi must be (6,)");
    Vector6d xi;
    std::memcpy(xi.data(), xi_arr.data(), 6 * sizeof(double));
    Matrix4d T = se3_exp(xi);
    Matrix3d R = T.block<3,3>(0,0);
    auto R_out = py::array_t<double>({3, 3});
    double* rp = R_out.mutable_data();
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) rp[i * 3 + j] = R(j, i);  // row-major for numpy
    auto t_out = py::array_t<double>(3);
    std::memcpy(t_out.mutable_data(), T.block<3,1>(0,3).data(), 3 * sizeof(double));
    return py::make_tuple(R_out, t_out);
}

py::array_t<double> py_se3_log(py::array_t<double, py::array::c_style | py::array::forcecast> T_arr) {
    if (T_arr.size() != 16) throw std::runtime_error("se3_log: T must be (4,4)");
    Matrix4d T;
    Eigen::Map<RowMajorMatXd> T_map((double*)T_arr.data(), 4, 4);
    T = T_map;
    Vector6d xi = se3_log(T);
    auto out = py::array_t<double>(6);
    std::memcpy(out.mutable_data(), xi.data(), 6 * sizeof(double));
    return out;
}

py::array_t<double> py_se3_inverse(py::array_t<double, py::array::c_style | py::array::forcecast> T_arr) {
    if (T_arr.size() != 16) throw std::runtime_error("se3_inverse: T must be (4,4)");
    Matrix4d T;
    Eigen::Map<RowMajorMatXd> T_map((double*)T_arr.data(), 4, 4);
    T = T_map;  // row-major read
    Matrix4d Tinv = se3_inverse(T);
    auto out = py::array_t<double>({4, 4});
    Eigen::Map<RowMajorMatXd> out_map(out.mutable_data(), 4, 4);
    out_map = Tinv;
    return out;
}

py::array_t<double> py_se3_compose(
    py::array_t<double, py::array::c_style | py::array::forcecast> T1_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> T2_arr
) {
    if (T1_arr.size() != 16 || T2_arr.size() != 16) throw std::runtime_error("se3_compose: T1, T2 must be (4,4)");
    Matrix4d T1, T2;
    Eigen::Map<RowMajorMatXd> T1_map((double*)T1_arr.data(), 4, 4);
    Eigen::Map<RowMajorMatXd> T2_map((double*)T2_arr.data(), 4, 4);
    T1 = T1_map; T2 = T2_map;
    Matrix4d T = se3_compose(T1, T2);
    auto out = py::array_t<double>({4, 4});
    Eigen::Map<RowMajorMatXd> out_map(out.mutable_data(), 4, 4);
    out_map = T;
    return out;
}

py::array_t<double> py_transform_point(
    py::array_t<double, py::array::c_style | py::array::forcecast> T_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> p_arr
) {
    if (T_arr.size() != 16) throw std::runtime_error("transform_point: T must be (4,4)");
    Matrix4d T;
    Eigen::Map<RowMajorMatXd> T_map((double*)T_arr.data(), 4, 4);
    T = T_map;
    py::buffer_info pb = p_arr.request();
    if (pb.ndim == 1 && pb.shape[0] == 3) {
        Vector3d p;
        std::memcpy(p.data(), p_arr.data(), 3 * sizeof(double));
        Vector3d q = transform_point_single(T, p);
        auto out = py::array_t<double>(3);
        std::memcpy(out.mutable_data(), q.data(), 3 * sizeof(double));
        return out;
    }
    if (pb.ndim == 2 && pb.shape[1] >= 3) {
        size_t N = pb.shape[0];
        std::vector<py::ssize_t> shape_out = { static_cast<py::ssize_t>(N), 3 };
        auto out = py::array_t<double>(shape_out);
        const double* src = static_cast<const double*>(pb.ptr);
        double* dst = out.mutable_data();
        size_t stride = pb.strides[0] / sizeof(double);
        for (size_t i = 0; i < N; ++i) {
            Vector3d p(src[i * stride], src[i * stride + 1], src[i * stride + 2]);
            Vector3d q = transform_point_single(T, p);
            dst[i * 3] = q(0); dst[i * 3 + 1] = q(1); dst[i * 3 + 2] = q(2);
        }
        return out;
    }
    throw std::runtime_error("transform_point: p must be (3,) or (N,3)");
}

py::array_t<double> py_adjoint_se3(py::array_t<double, py::array::c_style | py::array::forcecast> T_arr) {
    if (T_arr.size() != 16) throw std::runtime_error("adjoint_se3: T must be (4,4)");
    Matrix4d T;
    Eigen::Map<RowMajorMatXd> T_map((double*)T_arr.data(), 4, 4);
    T = T_map;
    Eigen::Matrix<double, 6, 6> Ad = adjoint_se3(T);
    auto out = py::array_t<double>({6, 6});
    Eigen::Map<RowMajorMatXd> Ad_out(out.mutable_data(), 6, 6);
    Ad_out = Ad;
    return out;
}

py::array_t<double> py_se3_to_matrix(
    py::array_t<double, py::array::c_style | py::array::forcecast> R_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_arr
) {
    if (R_arr.size() != 9 || t_arr.size() != 3) throw std::runtime_error("se3_to_matrix: R (3,3), t (3,)");
    Matrix3d R;
    const double* rp = R_arr.data();
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) R(j, i) = rp[i * 3 + j];  // row-major read
    Vector3d t;
    std::memcpy(t.data(), t_arr.data(), 3 * sizeof(double));
    Matrix4d T = Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
    auto out = py::array_t<double>({4, 4});
    Eigen::Map<RowMajorMatXd> out_map(out.mutable_data(), 4, 4);
    out_map = T;
    return out;
}

py::tuple py_matrix_to_se3(py::array_t<double, py::array::c_style | py::array::forcecast> T_arr) {
    if (T_arr.size() != 16) throw std::runtime_error("matrix_to_se3: T must be (4,4)");
    Matrix4d T;
    Eigen::Map<RowMajorMatXd> T_map((double*)T_arr.data(), 4, 4);
    T = T_map;
    Matrix3d R = T.block<3,3>(0,0);
    Vector3d t = T.block<3,1>(0,3);
    auto R_out = py::array_t<double>({3, 3});
    double* rp = R_out.mutable_data();
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) rp[i * 3 + j] = R(j, i);
    auto t_out = py::array_t<double>(3);
    std::memcpy(t_out.mutable_data(), t.data(), 3 * sizeof(double));
    return py::make_tuple(R_out, t_out);
}


// ─── pybind11 module ──────────────────────────────────────────────────────────

PYBIND11_MODULE(iv_gicp_core, m) {
    m.doc() = "IV-GICP C++ core: Eigen-based GN ICP loop (full registration in one call).";

    py::class_<RegistrationSession>(m, "RegistrationSession")
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
             py::arg("means_3d"),
             "Create a session that caches the target KDTree. Pass means_3d (N,3). Reuse for multiple register calls with the same target.")
        .def("register", &RegistrationSession::register_,
             py::arg("src_xyz"), py::arg("src_int"),
             py::arg("tgt_means"), py::arg("tgt_prec"), py::arg("tgt_grads"),
             py::arg("T_init"),
             py::arg("max_corr_dist") = 2.0,
             py::arg("alpha") = 0.1,
             py::arg("max_iter") = 30,
             py::arg("huber_delta") = 1.0,
             py::arg("min_valid") = 6,
             py::arg("use_fim_weight") = false,
             py::arg("max_source_points") = 0,
             py::arg("gm_scale") = 0.0,
             py::arg("fim_gate_ratio") = 0.0,
             py::arg("trim_ratio") = 0.0,
             py::arg("use_mscs") = false,
             py::arg("mscs_kappa_max") = 100.0,
             py::arg("v_min_prev") = py::array_t<double>(6),
             "Register using cached KDTree. Returns dict with T, H, v_min, n_valid, n_mscs_used, iterations, converged, tree_build_ms, gn_loop_ms.");

    m.def("icp_register", &icp_register,
        py::arg("src_xyz"), py::arg("src_int"),
        py::arg("tgt_means"), py::arg("tgt_prec"), py::arg("tgt_grads"),
        py::arg("kdtree_pts"), py::arg("T_init"),
        py::arg("max_corr_dist") = 2.0,
        py::arg("alpha") = 0.1,
        py::arg("max_iter") = 30,
        py::arg("huber_delta") = 1.0,
        py::arg("min_valid") = 6,
        py::arg("use_fim_weight") = false,
        py::arg("max_source_points") = 0,
        py::arg("gm_scale") = 0.0,
        py::arg("fim_gate_ratio") = 0.0,
        py::arg("trim_ratio") = 0.0,
        py::arg("use_mscs") = false,
        py::arg("mscs_kappa_max") = 100.0,
        py::arg("v_min_prev") = py::array_t<double>(6),
        R"doc(
Full IV-GICP Gauss-Newton registration loop in C++/Eigen.

Args:
    src_xyz:        (M, 3) float64 — source points (NOT pre-transformed)
    src_int:        (M,)   float64 — source intensities [0,1]
    tgt_means:      (N, 4) float64 — target voxel [x,y,z,α·I]
    tgt_prec:       (N, 4, 4) float64 — precision matrices Ω (C⁻¹)
    tgt_grads:      (N, 3) float64 — intensity gradients ∇μ_I
    kdtree_pts:     (N, 3) float64 — target voxel centers for KDTree (= tgt_means[:,:3])
    T_init:         (4, 4) float64 — initial pose (world←sensor)
    max_corr_dist:  max correspondence distance [m]
    alpha:          intensity weight (0 = geometry-only GICP)
    max_iter:       max GN iterations
    huber_delta:    Huber threshold on 4D Mahalanobis residual r = sqrt(d^T Omega d) (0 = disabled)
    min_valid:      min correspondences to continue
    use_mscs:       C4 MSCS — stop accumulating when λ_min(H) ≥ λ_max/mscs_kappa_max
    mscs_kappa_max: target condition number for MSCS stopping criterion (default 100)
    v_min_prev:     (6,) float64 — min eigenvector from previous frame (warm-start for MSCS sorting)

Returns:
    dict with keys:
        T            (4, 4) float64 — refined pose
        H            (6, 6) float64 — Hessian from last iteration (Fisher proxy / diagnostics)
        v_min        (6,)   float64 — min eigenvector of H (for next frame's MSCS warm-start)
        n_valid      int — valid correspondences in last iteration
        n_mscs_used  int — correspondences used by MSCS (0 if use_mscs=False)
        iterations   int — actual iterations run
        converged    bool — whether dx.norm() < 1e-6
        tree_build_ms float — KDTree build time [ms]
        gn_loop_ms   float — GN loop time [ms]
)doc");

    m.def("voxel_downsample", &voxel_downsample,
        py::arg("points"), py::arg("intensities"), py::arg("voxel_size"),
        "Voxel-grid downsampling: centroid per voxel. Returns (points_out, intensities_out).");

    m.def("se3_exp", &py_se3_exp, py::arg("xi"), "SE(3) exponential. xi (6,) -> (R 3x3, t 3).");
    m.def("se3_log", &py_se3_log, py::arg("T"), "SE(3) logarithm. T (4,4) -> xi (6,).");
    m.def("se3_inverse", &py_se3_inverse, py::arg("T"), "SE(3) inverse.");
    m.def("se3_compose", &py_se3_compose, py::arg("T1"), py::arg("T2"), "SE(3) compose T1*T2.");
    m.def("transform_point", &py_transform_point, py::arg("T"), py::arg("p"), "Transform point(s) by T.");
    m.def("adjoint_se3", &py_adjoint_se3, py::arg("T"), "Adjoint 6x6 of SE(3).");
    m.def("se3_to_matrix", &py_se3_to_matrix, py::arg("R"), py::arg("t"), "Build 4x4 from R,t.");
    m.def("matrix_to_se3", &py_matrix_to_se3, py::arg("T"), "Extract R,t from 4x4.");

    m.def("set_num_threads", [](int n) { omp_set_num_threads(n); },
          py::arg("n"), "Set OpenMP thread count for GN loop.");
    m.def("get_num_threads", []() { return omp_get_max_threads(); },
          "Get current OpenMP max thread count.");
}

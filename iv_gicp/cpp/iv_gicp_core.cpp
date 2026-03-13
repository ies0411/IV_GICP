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
#include <cstring>
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
    bool use_fim_weight = false
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

    // C1 FIM weight: per-valid storage (used only when use_fim_weight)
    using Mat36 = Eigen::Matrix<double, 3, 6>;
    std::vector<Eigen::Vector4d> valid_d;
    std::vector<Mat36> valid_J_xyz;
    std::vector<Matrix4d44> valid_Omega;
    std::vector<Matrix4x6d> valid_J;
    std::vector<double> valid_w_huber;

    for (iter = 0; iter < max_iter; ++iter) {
        Matrix3d R = T.block<3,3>(0,0);
        Vector3d t = T.block<3,1>(0,3);

        Matrix6d H = Matrix6d::Zero();
        Vector6d b = Vector6d::Zero();
        int n_valid = 0;

        if (use_fim_weight) {
            // C1 path: collect valid data (parallel over M), compute I_G and w_fim, then accumulate weighted H, b
            valid_d.clear();
            valid_J_xyz.clear();
            valid_Omega.clear();
            valid_J.clear();
            valid_w_huber.clear();
            Matrix6d I_G = Matrix6d::Zero();

            #pragma omp parallel
            {
                std::vector<Eigen::Vector4d> valid_d_loc;
                std::vector<Mat36> valid_J_xyz_loc;
                std::vector<Matrix4d44> valid_Omega_loc;
                std::vector<Matrix4x6d> valid_J_loc;
                std::vector<double> valid_w_huber_loc;
                Matrix6d I_G_loc = Matrix6d::Zero();
                std::vector<size_t> t_idx(1);
                std::vector<double> t_dsq(1);

                #pragma omp for nowait
                for (int i = 0; i < M; ++i) {
                    Vector3d p_s(src_xyz[i*3], src_xyz[i*3+1], src_xyz[i*3+2]);
                    Vector3d q = R * p_s + t;
                    nanoflann::KNNResultSet<double> rs(1);
                    rs.init(t_idx.data(), t_dsq.data());
                    double qd[3] = {q[0], q[1], q[2]};
                    tree.findNeighbors(rs, qd, nanoflann::SearchParameters());
                    if (t_dsq[0] > max_dist_sq) continue;
                    int j = (int)t_idx[0];

                    Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>> Omega(tgt_p + j*16);
                    Eigen::Map<const Eigen::Vector4d> mu_t(tgt_m + j*4);
                    Eigen::Map<const Vector3d> grad_t(tgt_g + j*3);
                    Eigen::Vector4d d;
                    d.head<3>() = q - mu_t.head<3>();
                    d[3]        = alpha * src_int[i] - mu_t[3];

                    Mat36 J_xyz;
                    J_xyz.setZero();
                    J_xyz(0,1)=q[2]; J_xyz(0,2)=-q[1]; J_xyz(1,0)=-q[2]; J_xyz(1,2)=q[0];
                    J_xyz(2,0)=q[1]; J_xyz(2,1)=-q[0];
                    J_xyz(0,3)=1.0; J_xyz(1,4)=1.0; J_xyz(2,5)=1.0;

                    Matrix4x6d J;
                    J.topRows<3>() = J_xyz;
                    J.row(3) = -alpha * (grad_t.transpose() * J_xyz);

                    double r_sq = d.transpose() * Omega * d;
                    double r = std::sqrt(std::max(r_sq, 0.0));
                    double w = 1.0;
                    if (huber_delta > 0.0 && r > huber_delta) w = huber_delta / (r + 1e-9);

                    valid_d_loc.push_back(d);
                    valid_J_xyz_loc.push_back(J_xyz);
                    valid_Omega_loc.push_back(Omega);
                    valid_J_loc.push_back(J);
                    valid_w_huber_loc.push_back(w);
                    Matrix3d Og = Omega.block<3,3>(0,0);
                    I_G_loc += J_xyz.transpose() * Og * J_xyz;
                }
                #pragma omp critical
                {
                    for (size_t k = 0; k < valid_d_loc.size(); ++k) {
                        valid_d.push_back(valid_d_loc[k]);
                        valid_J_xyz.push_back(valid_J_xyz_loc[k]);
                        valid_Omega.push_back(valid_Omega_loc[k]);
                        valid_J.push_back(valid_J_loc[k]);
                        valid_w_huber.push_back(valid_w_huber_loc[k]);
                    }
                    I_G += I_G_loc;
                }
            }
            n_valid = (int)valid_d.size();
            if (n_valid < min_valid) break;

            // FIM weight: v = min eigenvector of I_G, w_fim_i = v' H_all_i v, normalize
            Eigen::SelfAdjointEigenSolver<Matrix6d> es(I_G);
            Vector6d v = es.eigenvectors().col(0);
            double sum_w_fim = 0.0;
            std::vector<double> w_fim(n_valid, 1.0);
            for (int m = 0; m < n_valid; ++m) {
                Matrix6d H_all = valid_J[m].transpose() * valid_Omega[m] * valid_J[m];
                double wm = std::max(v.dot(H_all * v), 1e-12);
                w_fim[m] = wm;
                sum_w_fim += wm;
            }
            double mean_w = sum_w_fim / (n_valid + 1e-9);
            for (int m = 0; m < n_valid; ++m) {
                double w = valid_w_huber[m] * (w_fim[m] / mean_w);
                Eigen::Matrix<double, 6, 4> JtO = w * valid_J[m].transpose() * valid_Omega[m];
                H += JtO * valid_J[m];
                b += JtO * valid_d[m];
            }
        } else {
            #pragma omp parallel
            {
                Matrix6d H_loc = Matrix6d::Zero();
                Vector6d b_loc = Vector6d::Zero();
                int n_loc = 0;
                std::vector<size_t> t_idx(1);
                std::vector<double> t_dsq(1);

                #pragma omp for nowait
                for (int i = 0; i < M; ++i) {
                    Vector3d p_s(src_xyz[i*3], src_xyz[i*3+1], src_xyz[i*3+2]);
                    Vector3d q = R * p_s + t;
                    nanoflann::KNNResultSet<double> rs(1);
                    rs.init(t_idx.data(), t_dsq.data());
                    double qd[3] = {q[0], q[1], q[2]};
                    tree.findNeighbors(rs, qd, nanoflann::SearchParameters());
                    if (t_dsq[0] > max_dist_sq) continue;
                    int j = (int)t_idx[0];

                    Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>> Omega(tgt_p + j*16);
                    Eigen::Map<const Eigen::Vector4d> mu_t(tgt_m + j*4);
                    Eigen::Map<const Vector3d> grad_t(tgt_g + j*3);
                    Eigen::Vector4d d;
                    d.head<3>() = q - mu_t.head<3>();
                    d[3]        = alpha * src_int[i] - mu_t[3];

                    Eigen::Matrix<double, 3, 6> J_xyz;
                    J_xyz.setZero();
                    J_xyz(0,1)=q[2]; J_xyz(0,2)=-q[1]; J_xyz(1,0)=-q[2]; J_xyz(1,2)=q[0];
                    J_xyz(2,0)=q[1]; J_xyz(2,1)=-q[0];
                    J_xyz(0,3)=1.0; J_xyz(1,4)=1.0; J_xyz(2,5)=1.0;

                    Matrix4x6d J;
                    J.topRows<3>() = J_xyz;
                    J.row(3) = -alpha * (grad_t.transpose() * J_xyz);

                    double r_sq = d.transpose() * Omega * d;
                    double r = std::sqrt(std::max(r_sq, 0.0));
                    double w = 1.0;
                    if (huber_delta > 0.0 && r > huber_delta) w = huber_delta / (r + 1e-9);

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
    huber_delta:   Huber threshold on 4D Mahalanobis residual r = sqrt(d^T Omega d) (0 = disabled)
    min_valid:     min correspondences to continue

Returns:
    dict with keys:
        T          (4, 4) float64 — refined pose
        H          (6, 6) float64 — Hessian from last iteration (for C3 window smoothing)
        n_valid    int — valid correspondences in last iteration
        iterations int — actual iterations run
        converged  bool — whether dx.norm() < 1e-6
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
}

/**
 * iv_gicp_map.cpp  (v2)
 *
 * Additions over v1:
 *   [A] Cached KDTree in VoxelMap — reused for query_sigma (replaces Python FastKDTree.query)
 *   [B] query_sigma()             — C++ median inlier distance (adaptive sigma update)
 *   [C] downsample_and_filter()  — O(N) hash-based source prefilter (replaces Python voxel_downsample)
 *
 * Together with the v1 VoxelMap these eliminate all Python map/prefilter bottlenecks.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "nanoflann.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <unordered_map>
#include <set>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <omp.h>

namespace py = pybind11;
using Matrix3d = Eigen::Matrix3d;

// ── Voxel key & hash ───────────────────────────────────────────────────────────

struct VoxelKey {
    int32_t x, y, z;
    bool operator==(const VoxelKey& o) const { return x==o.x && y==o.y && z==o.z; }
    bool operator<(const VoxelKey& o) const {
        if (x != o.x) return x < o.x;
        if (y != o.y) return y < o.y;
        return z < o.z;
    }
};
struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const noexcept {
        return (size_t)(k.x*73856093) ^ (size_t)(k.y*19349663) ^ (size_t)(k.z*83492791);
    }
};

// Compact key for downsample_and_filter (same hash but separate struct)
struct DSKey { int32_t x,y,z; bool operator==(const DSKey& o) const{return x==o.x&&y==o.y&&z==o.z;} };
struct DSKeyHash { size_t operator()(const DSKey& k) const noexcept {
    return (size_t)(k.x*73856093)^(size_t)(k.y*19349663)^(size_t)(k.z*83492791); } };

// ── Per-voxel Welford state ─────────────────────────────────────────────────

struct VoxelState {
    int    n          = 0;
    double mean[3]    = {0,0,0};
    double M2[9]      = {0};   // 3×3 row-major
    double mean_i     = 0.0;
    double M2_i       = 0.0;
    int    last_frame = -1;
    int    n_frames   = 0;     // distinct frames that contributed (for TSG)
};

// ── nanoflann cloud adapter ───────────────────────────────────────────────────

struct FlatCloud {
    const double* ptr;
    size_t N;
    inline size_t kdtree_get_point_count() const { return N; }
    inline double kdtree_get_pt(size_t idx, size_t dim) const { return ptr[idx*3+dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};
using VoxelKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, FlatCloud>, FlatCloud, 3>;


// ── VoxelMap ───────────────────────────────────────────────────────────────────

class VoxelMap {
public:
    // max_n_per_voxel: EMA forgetting factor cap (0 = unlimited Welford, KISS uses 20).
    // When n >= max_n, mean/M2 are updated with a sliding window EMA so stale
    // observations are exponentially down-weighted. This prevents "voxel freezing"
    // in long sequences where n→∞ makes new observations contribute ~0.
    VoxelMap(double voxel_size, int min_points, int max_n_per_voxel = 100)
        : vs_(voxel_size), min_pts_(min_points), max_n_(max_n_per_voxel) {}

    // ── insert_frame: O(N) Welford per-point ─────────────────────────────────
    void insert_frame(
        py::array_t<double, py::array::c_style|py::array::forcecast> xyz_a,
        py::array_t<double, py::array::c_style|py::array::forcecast> int_a,
        int frame_id)
    {
        auto xb = xyz_a.request(), ib = int_a.request();
        const int N = (int)xb.shape[0];
        if (N == 0) return;
        const double* xyz  = (double*)xb.ptr;
        const double* ints = (double*)ib.ptr;
        const double inv = 1.0 / vs_;

        for (int i = 0; i < N; ++i) {
            double px = xyz[i*3], py = xyz[i*3+1], pz = xyz[i*3+2];
            VoxelKey key{ (int32_t)std::floor(px*inv),
                          (int32_t)std::floor(py*inv),
                          (int32_t)std::floor(pz*inv) };
            auto& s = voxels_[key];
            if (frame_id != s.last_frame) s.n_frames++;  // count distinct frames
            s.last_frame = frame_id;
            // EMA forgetting: when n >= max_n, use sliding window EMA so stale
            // observations are down-weighted. Prevents "voxel freezing" over long
            // sequences where n→∞ would make new observations contribute ~1/n ≈ 0.
            // Standard Welford when max_n_==0 (unlimited) or n < max_n_.
            const bool use_ema = (max_n_ > 0 && s.n >= max_n_);
            if (!use_ema) s.n++;
            const double w = use_ema ? (1.0 / max_n_) : (1.0 / s.n);
            double d0=px-s.mean[0], d1=py-s.mean[1], d2=pz-s.mean[2];
            s.mean[0]+=d0*w; s.mean[1]+=d1*w; s.mean[2]+=d2*w;
            double e0=px-s.mean[0], e1=py-s.mean[1], e2=pz-s.mean[2];
            double dv[3]={d0,d1,d2}, ev[3]={e0,e1,e2};
            if (use_ema) {
                // Scale M2 before adding: M2_new = M2*(1-w) + d⊗e
                // keeps M2 ≈ (max_n-1)*cov in steady state
                const double decay = 1.0 - w;
                for(int r=0;r<3;++r) for(int c=0;c<3;++c)
                    s.M2[r*3+c] = s.M2[r*3+c]*decay + dv[r]*ev[c];
                double di=ints[i]-s.mean_i;
                s.mean_i+=di*w;
                s.M2_i = s.M2_i*decay + di*(ints[i]-s.mean_i);
            } else {
                for(int r=0;r<3;++r) for(int c=0;c<3;++c) s.M2[r*3+c]+=dv[r]*ev[c];
                double di=ints[i]-s.mean_i;
                s.mean_i+=di/s.n; s.M2_i+=di*(ints[i]-s.mean_i);
            }
        }
    }

    // ── evict_before ─────────────────────────────────────────────────────────
    int evict_before(int frame_id) {
        int n=0;
        for(auto it=voxels_.begin();it!=voxels_.end();) {
            if(it->second.last_frame < frame_id) { it=voxels_.erase(it); ++n; }
            else ++it;
        }
        return n;
    }

    // ── evict_far_from: spatial eviction (KISS-ICP style) ────────────────────
    // Remove voxels whose mean position is farther than max_dist from (cx, cy, cz).
    // This keeps the map dense near the robot regardless of frame age, matching
    // KISS-ICP's spatial eviction strategy which outperforms age-based eviction
    // in long-range sequences (the robot has moved far from old voxels).
    int evict_far_from(double cx, double cy, double cz, double max_dist) {
        int n=0;
        const double md2 = max_dist * max_dist;
        for(auto it=voxels_.begin();it!=voxels_.end();) {
            const VoxelState& s = it->second;
            if(s.n < 1) { ++it; continue; }  // skip empty (shouldn't happen)
            double dx=s.mean[0]-cx, dy=s.mean[1]-cy, dz=s.mean[2]-cz;
            if(dx*dx + dy*dy + dz*dz > md2) { it=voxels_.erase(it); ++n; }
            else ++it;
        }
        return n;
    }

    // ── evict_incoherent: dynamic object rejection via Mahalanobis coherence ─
    // Called BEFORE insert_frame with current scan in world frame.
    // For each source point, check Mahalanobis distance to the map voxel it lands in.
    // If a voxel has enough source points but <min_inlier_ratio of them are within
    // mahal_sq_threshold σ², the voxel's distribution is stale (dynamic object moved
    // through) — evict it before inserting so we don't propagate corrupted covariance.
    //
    // Args:
    //   xyz_world:          (N,3) float64 — current scan in world frame
    //   mahal_sq_threshold: Mahalanobis² threshold for "inlier" (default 9.0 = 3σ²)
    //   min_inlier_ratio:   evict if n_inliers/n_hits < this (default 0.3)
    //   min_test_points:    need at least this many src pts in voxel to test (default 3)
    //
    // Returns: number of evicted voxels
    int evict_incoherent(
        py::array_t<double, py::array::c_style|py::array::forcecast> xyz_world_a,
        double mahal_sq_threshold = 9.0,
        double min_inlier_ratio   = 0.3,
        int    min_test_points    = 3)
    {
        auto xb = xyz_world_a.request();
        const int N = (int)xb.shape[0];
        if (N == 0 || voxels_.empty()) return 0;
        const double* xyz = (double*)xb.ptr;
        const double inv = 1.0 / vs_;
        const double reg = 1e-6;
        const Matrix3d I3 = Matrix3d::Identity();

        // Per-voxel test state: cached precision + hit/inlier counts
        struct VoxelTest {
            Matrix3d Omega;
            double mu[3];
            int n_hits    = 0;
            int n_inliers = 0;
            bool ready    = false;
        };
        std::unordered_map<VoxelKey, VoxelTest, VoxelKeyHash> tests;
        tests.reserve(512);

        for (int i = 0; i < N; ++i) {
            double px = xyz[i*3], py = xyz[i*3+1], pz = xyz[i*3+2];
            VoxelKey key{ (int32_t)std::floor(px*inv),
                          (int32_t)std::floor(py*inv),
                          (int32_t)std::floor(pz*inv) };
            auto vit = voxels_.find(key);
            if (vit == voxels_.end()) continue;
            const VoxelState& s = vit->second;
            if (s.n < min_pts_) continue;

            auto& t = tests[key];
            if (!t.ready) {
                // Compute precision matrix once per voxel
                const int n_eff = (max_n_ > 0 && s.n >= max_n_) ? max_n_ : s.n;
                int ns = std::max(n_eff - 1, 1);
                Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> M2m(s.M2);
                Matrix3d Sig = M2m / ns + reg * I3;
                t.Omega = Sig.ldlt().solve(I3);
                t.mu[0] = s.mean[0]; t.mu[1] = s.mean[1]; t.mu[2] = s.mean[2];
                t.ready = true;
            }
            t.n_hits++;
            Eigen::Vector3d dp(px - t.mu[0], py - t.mu[1], pz - t.mu[2]);
            if (dp.dot(t.Omega * dp) <= mahal_sq_threshold) t.n_inliers++;
        }

        int n_evicted = 0;
        for (const auto& kv : tests) {
            if (kv.second.n_hits >= min_test_points &&
                (double)kv.second.n_inliers / kv.second.n_hits < min_inlier_ratio) {
                voxels_.erase(kv.first);
                ++n_evicted;
            }
        }
        return n_evicted;
    }

    // ── evict_ghost_raycast: free-space consistency check via DDA ray marching ─
    // LiDAR rays travel in straight lines. If a scan point p is observed, the
    // entire path sensor→p must be free space. Any map voxel along that path is
    // a ghost (stale dynamic object distribution) and should be removed.
    //
    // For each scan point, march a DDA ray from the sensor origin to the point.
    // Intermediate voxels get pass_count++; the endpoint voxel gets hit_count++.
    // A voxel with pass_count/(pass_count+hit_count) > ghost_threshold is evicted.
    //
    // Args:
    //   sx,sy,sz:         sensor origin in world frame
    //   scan_xyz:         (N,3) float64 — current scan points in world frame
    //   ghost_threshold:  evict if pass_ratio > this (default 0.5)
    //   min_pass_through: need ≥ this many pass-through rays (default 3)
    //
    // Returns: number of evicted voxels
    int evict_ghost_raycast(
        double sx, double sy, double sz,
        py::array_t<double, py::array::c_style|py::array::forcecast> scan_xyz_a,
        double ghost_threshold = 0.5,
        int    min_pass_through = 3)
    {
        auto xb = scan_xyz_a.request();
        const int N = (int)xb.shape[0];
        if (N == 0 || voxels_.empty()) return 0;
        const double* xyz = (double*)xb.ptr;
        const double inv = 1.0 / vs_;

        // Per-voxel ray evidence counters
        std::unordered_map<VoxelKey, std::pair<int,int>, VoxelKeyHash> counts;
        // counts[key] = {pass_count, hit_count}
        counts.reserve(1024);

        for (int i = 0; i < N; ++i) {
            double ex = xyz[i*3], ey = xyz[i*3+1], ez = xyz[i*3+2];

            // Start and end voxel coords
            int32_t vx = (int32_t)std::floor(sx*inv);
            int32_t vy = (int32_t)std::floor(sy*inv);
            int32_t vz = (int32_t)std::floor(sz*inv);
            int32_t ex_v = (int32_t)std::floor(ex*inv);
            int32_t ey_v = (int32_t)std::floor(ey*inv);
            int32_t ez_v = (int32_t)std::floor(ez*inv);

            // DDA setup (Amanatides & Woo 1987)
            double dx = ex - sx, dy = ey - sy, dz = ez - sz;
            double len = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (len < 1e-9) { counts[{ex_v,ey_v,ez_v}].second++; continue; }
            double rx = dx/len, ry = dy/len, rz = dz/len;

            int stepX = (rx > 0) ? 1 : -1;
            int stepY = (ry > 0) ? 1 : -1;
            int stepZ = (rz > 0) ? 1 : -1;

            double tDeltaX = (std::abs(rx) > 1e-12) ? (vs_ / std::abs(rx)) : 1e30;
            double tDeltaY = (std::abs(ry) > 1e-12) ? (vs_ / std::abs(ry)) : 1e30;
            double tDeltaZ = (std::abs(rz) > 1e-12) ? (vs_ / std::abs(rz)) : 1e30;

            // t at first voxel boundary
            auto nextBound = [&](double s_coord, int step, double r_dir) -> double {
                if (std::abs(r_dir) < 1e-12) return 1e30;
                double voxel_pos = std::floor(s_coord * inv) * vs_;
                double boundary = (step > 0) ? (voxel_pos + vs_) : voxel_pos;
                return (boundary - s_coord) / r_dir;
            };
            double tMaxX = nextBound(sx, stepX, rx);
            double tMaxY = nextBound(sy, stepY, ry);
            double tMaxZ = nextBound(sz, stepZ, rz);

            // March ray — stop at endpoint voxel or when t exceeds ray length
            const int max_steps = (int)(len / vs_) + 4;  // safety limit
            for (int s = 0; s < max_steps; ++s) {
                if (vx == ex_v && vy == ey_v && vz == ez_v) break;

                // Only mark as pass-through if this voxel exists in the map
                VoxelKey key{vx, vy, vz};
                if (voxels_.find(key) != voxels_.end())
                    counts[key].first++;  // pass_count

                // Advance to next voxel
                if (tMaxX < tMaxY && tMaxX < tMaxZ) {
                    vx += stepX; tMaxX += tDeltaX;
                } else if (tMaxY < tMaxZ) {
                    vy += stepY; tMaxY += tDeltaY;
                } else {
                    vz += stepZ; tMaxZ += tDeltaZ;
                }
            }
            // Endpoint voxel = hit
            counts[{ex_v,ey_v,ez_v}].second++;
        }

        // Evict voxels contradicted by ray evidence
        int n_evicted = 0;
        for (const auto& kv : counts) {
            int pass = kv.second.first;
            int hit  = kv.second.second;
            if (pass >= min_pass_through) {
                double ratio = (double)pass / (pass + hit + 1e-9);
                if (ratio > ghost_threshold) {
                    voxels_.erase(kv.first);
                    ++n_evicted;
                }
            }
        }
        return n_evicted;
    }

    int size() const { return (int)voxels_.size(); }

    // ── build_target_arrays: covariances + precisions + gradients ────────────
    /**
     * Builds all target arrays in one C++ pass, AND caches the voxel-center KDTree
     * for reuse by query_sigma() (adaptive correspondence distance update).
     *
     * Returns dict{"means_4d":(V,4), "prec":(V,4,4), "grads":(V,3), "means_3d":(V,3)}
     */
    py::dict build_target_arrays(double alpha, double source_sigma,
                                 int k_grad_nbrs, double count_reg_scale,
                                 bool use_entropy_alpha = false, double entropy_scale_c = 0.3,
                                 int stable_frames = 0,  // TSG: voxels with n_frames < stable_frames get isotropic blend
                                 int max_target_voxels = 0)  // >0: LOAM-style cap — keep top-K by observation count n
    {
        (void)use_entropy_alpha;
        (void)entropy_scale_c;
        const double eps=1e-6, eps_var=1e-4;
        const double vs2=vs_*vs_, ss2=source_sigma*source_sigma;
        const double cr2=count_reg_scale*count_reg_scale, a2=alpha*alpha;
        const Matrix3d I3 = Matrix3d::Identity();

        // 1. Collect valid voxels
        std::vector<const VoxelState*> valid;
        valid.reserve(voxels_.size());
        for(auto& kv:voxels_) if(kv.second.n>=min_pts_) valid.push_back(&kv.second);

        // 1b. Optional cap: keep voxels with largest n (strongest statistics) for registration + KD only.
        // Full map insert/eviction unchanged; this only affects the target arrays used by GN.
        if (max_target_voxels > 0 && (int)valid.size() > max_target_voxels) {
            std::nth_element(valid.begin(), valid.begin() + max_target_voxels, valid.end(),
                [](const VoxelState* a, const VoxelState* b) { return a->n > b->n; });
            valid.resize((size_t)max_target_voxels);
        }

        const int V = (int)valid.size();

        if (V == 0) {
            cached_V_ = 0; cached_tree_.reset(); cached_cloud_.reset();
            py::dict d;
            d["means_4d"]=py::array_t<double>(std::vector<ssize_t>{0,4});
            d["prec"]    =py::array_t<double>(std::vector<ssize_t>{0,4,4});
            d["grads"]   =py::array_t<double>(std::vector<ssize_t>{0,3});
            d["means_3d"]=py::array_t<double>(std::vector<ssize_t>{0,3});
            return d;
        }

        // 2. Allocate output arrays
        auto m4a  = py::array_t<double>(std::vector<ssize_t>{V,4});
        auto pra  = py::array_t<double>(std::vector<ssize_t>{V,4,4});
        auto gra  = py::array_t<double>(std::vector<ssize_t>{V,3});
        auto m3a  = py::array_t<double>(std::vector<ssize_t>{V,3});
        double* m4=(double*)m4a.request().ptr, *pr=(double*)pra.request().ptr;
        double* gr=(double*)gra.request().ptr, *m3=(double*)m3a.request().ptr;

        // 3. Compute means + precision matrices (OpenMP)
        #pragma omp parallel for schedule(static)
        for(int i=0;i<V;++i) {
            const VoxelState& s=*valid[i];
            // When EMA is active (n capped at max_n_), covariance denominator = max_n_-1.
            const int n_eff = (max_n_ > 0 && s.n >= max_n_) ? max_n_ : s.n;
            const int ns=std::max(n_eff-1,1);
            const double inv=1.0/ns;

            Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> M2m(s.M2);
            Matrix3d Sig = M2m*inv + (eps+ss2+cr2/std::max(s.n,1))*I3;
            Matrix3d Og  = Sig.ldlt().solve(I3);

            // TSG: temporal stability gating — newly-seen voxels (transient dynamic objects)
            // get their precision blended toward isotropic. beta → 1 as voxel matures.
            if (stable_frames > 0 && s.n_frames < stable_frames) {
                double beta = (double)s.n_frames / stable_frames;
                double iso_w = Og.trace() / 3.0;
                Og = beta * Og + (1.0 - beta) * iso_w * I3;
            }

            double vi=s.M2_i*inv;
            double gp=vi/(vs2+1e-9);
            double sq=std::min(std::max(a2/(gp+eps_var),1e-6),1e6);
            double oI=1.0/(sq+eps);

            m4[i*4+0]=s.mean[0]; m4[i*4+1]=s.mean[1];
            m4[i*4+2]=s.mean[2]; m4[i*4+3]=alpha*s.mean_i;
            m3[i*3+0]=s.mean[0]; m3[i*3+1]=s.mean[1]; m3[i*3+2]=s.mean[2];

            double* p=pr+i*16;
            for(int r=0;r<4;++r) for(int c=0;c<4;++c) p[r*4+c]=0.0;
            for(int r=0;r<3;++r) for(int c=0;c<3;++c) p[r*4+c]=Og(r,c);
            p[15]=oI;
            gr[i*3]=gr[i*3+1]=gr[i*3+2]=0.0;
        }

        // 4. Cache KDTree for query_sigma (build from m3 output buffer)
        //    Copy m3 into cached_m3_ so the pointer remains stable across calls.
        cached_m3_.assign(m3, m3+V*3);
        cached_cloud_ = std::make_shared<FlatCloud>(FlatCloud{cached_m3_.data(),(size_t)V});
        cached_tree_  = std::make_shared<VoxelKDTree>(3,*cached_cloud_,
                            nanoflann::KDTreeSingleIndexAdaptorParams(10));
        cached_tree_->buildIndex();
        cached_V_ = V;

        // 5. Intensity gradients: per-thread KDTree so parallel queries are thread-safe
        if(alpha>1e-9 && V>=4 && k_grad_nbrs>=3) {
            const int K=std::min(k_grad_nbrs+1,V);
            #pragma omp parallel
            {
                FlatCloud cloud{cached_m3_.data(), (size_t)V};
                VoxelKDTree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
                tree.buildIndex();
                std::vector<size_t> ni(K);
                std::vector<double> nd(K);
                #pragma omp for schedule(static)
                for(int i=0;i<V;++i) {
                    nanoflann::KNNResultSet<double> rs(K);
                    rs.init(ni.data(),nd.data());
                    double qd[3]={m3[i*3],m3[i*3+1],m3[i*3+2]};
                    tree.findNeighbors(rs,qd,nanoflann::SearchParameters());

                    const double Ii=valid[i]->mean_i;
                    Eigen::MatrixXd A(K-1,3); Eigen::VectorXd b(K-1);
                    int row=0;
                    for(int ki=0;ki<K&&row<K-1;++ki) {
                        int j=(int)ni[ki]; if(j==i) continue;
                        A(row,0)=m3[j*3]-m3[i*3]; A(row,1)=m3[j*3+1]-m3[i*3+1];
                        A(row,2)=m3[j*3+2]-m3[i*3+2]; b(row)=valid[j]->mean_i-Ii; ++row;
                    }
                    if(row<3) continue;
                    Eigen::MatrixXd Ar=A.topRows(row); Eigen::VectorXd br=b.head(row);
                    Eigen::Matrix3d AtA=Ar.transpose()*Ar; AtA.diagonal().array()+=1e-6;
                    Eigen::Vector3d g=AtA.ldlt().solve(Ar.transpose()*br);
                    gr[i*3]=g[0]; gr[i*3+1]=g[1]; gr[i*3+2]=g[2];
                }
            }
        }

        py::dict res;
        res["means_4d"]=m4a; res["prec"]=pra; res["grads"]=gra; res["means_3d"]=m3a;
        return res;
    }

    // ── query_sigma: adaptive sigma update via cached KDTree ─────────────────
    /**
     * Query the cached voxel-center KDTree for pts_world, return median inlier distance.
     * Replaces Python: dists,_ = _target_tree.query(pts_world,k=1); median(dists[mask])
     *
     * Returns max(median_inlier_dist, min_threshold). Returns min_threshold if no inliers.
     */
    double query_sigma(
        py::array_t<double, py::array::c_style|py::array::forcecast> pts_a,
        double max_dist,
        double min_threshold)
    {
        if(!cached_tree_ || cached_V_==0) return min_threshold;

        auto pb=pts_a.request();
        const int M=(int)pb.shape[0];
        if(M==0) return min_threshold;
        const double* pts=(double*)pb.ptr;
        const double max_d2=max_dist*max_dist;

        std::vector<double> inlier_dists;
        inlier_dists.reserve(M);
        for (int i = 0; i < M; ++i) {
            size_t idx; double d2;
            nanoflann::KNNResultSet<double> rs(1);
            rs.init(&idx, &d2);
            double qd[3] = {pts[i*3], pts[i*3+1], pts[i*3+2]};
            cached_tree_->findNeighbors(rs, qd, nanoflann::SearchParameters());
            if (d2 <= max_d2) inlier_dists.push_back(std::sqrt(d2));
        }
        if(inlier_dists.size()<5) return min_threshold;
        size_t mid=inlier_dists.size()/2;
        std::nth_element(inlier_dists.begin(),inlier_dists.begin()+mid,inlier_dists.end());
        return std::max(inlier_dists[mid], min_threshold);
    }

    /** Max condition number κ over all voxels (for auto_alpha: κ high → degenerate → use intensity). */
    double get_max_condition_number() const {
        const double reg = 1e-6;
        const Matrix3d I3 = Matrix3d::Identity();
        double max_k = 1.0;
        for (const auto& kv : voxels_) {
            const VoxelState& s = kv.second;
            if (s.n < min_pts_) continue;
            int ns = std::max(s.n - 1, 1);
            Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> M2m(s.M2);
            Matrix3d Sig = M2m / ns + reg * I3;
            Eigen::SelfAdjointEigenSolver<Matrix3d> es(Sig);
            const auto& ev = es.eigenvalues();
            double lam_min = std::max(ev.minCoeff(), 1e-12);
            double lam_max = std::max(ev.maxCoeff(), 1e-12);
            double k = lam_max / lam_min;
            if (k > max_k) max_k = k;
        }
        return max_k;
    }

private:
    double vs_;
    int    min_pts_;
    int    max_n_;   // EMA cap (0 = unlimited)
    std::unordered_map<VoxelKey, VoxelState, VoxelKeyHash> voxels_;

    // Cached KDTree (built in build_target_arrays, reused by query_sigma)
    std::vector<double>              cached_m3_;
    std::shared_ptr<FlatCloud>       cached_cloud_;
    std::shared_ptr<VoxelKDTree>     cached_tree_;
    int                              cached_V_ = 0;
};



// ── downsample_and_filter: O(N) prefilter + voxel centroid ────────────────────
/**
 * Combined range filter + voxel-grid downsampling + intensity normalization.
 * Replaces Python voxel_downsample() (O(N log N) lexsort) with O(N) hash.
 *
 * Args:
 *   xyz:        (N, 3+) float64  — input point cloud
 *   ints:       (N,)    float64  — intensities
 *   voxel_size: float            — source voxel grid size
 *   min_range:  float            — min range filter
 *   max_range:  float            — max range filter
 *
 * Returns: (pts_ds, ints_ds) — downsampled (M, 3) and (M,) arrays
 */
struct DSState {
    double sx=0,sy=0,sz=0,si=0; int n=0;
};

py::tuple downsample_and_filter(
    py::array_t<double, py::array::c_style|py::array::forcecast> xyz_a,
    py::array_t<double, py::array::c_style|py::array::forcecast> int_a,
    double voxel_size,
    double min_range,
    double max_range)
{
    auto xb=xyz_a.request(), ib=int_a.request();
    const int N=(int)xb.shape[0];
    const double* xyz=(double*)xb.ptr;
    const double* ints=(double*)ib.ptr;
    const double inv=1.0/voxel_size;
    const double r2_min=min_range*min_range, r2_max=max_range*max_range;

    std::unordered_map<DSKey,DSState,DSKeyHash> voxels;
    voxels.reserve(N/4);

    for(int i=0;i<N;++i) {
        double x=xyz[i*3],y=xyz[i*3+1],z=xyz[i*3+2];
        double r2=x*x+y*y+z*z;
        if(r2<r2_min||r2>r2_max) continue;
        DSKey key{(int32_t)std::floor(x*inv),(int32_t)std::floor(y*inv),(int32_t)std::floor(z*inv)};
        auto& s=voxels[key];
        s.sx+=x; s.sy+=y; s.sz+=z; s.si+=ints[i]; s.n++;
    }

    const int M=(int)voxels.size();
    auto pts_out = py::array_t<double>(std::vector<ssize_t>{M,3});
    auto int_out = py::array_t<double>(std::vector<ssize_t>{M});
    double* po=(double*)pts_out.request().ptr;
    double* io=(double*)int_out.request().ptr;

    int j=0;
    double p99_max=0.0;
    for(auto& kv:voxels) {
        double inv_n=1.0/kv.second.n;
        po[j*3+0]=kv.second.sx*inv_n;
        po[j*3+1]=kv.second.sy*inv_n;
        po[j*3+2]=kv.second.sz*inv_n;
        io[j]=kv.second.si*inv_n;
        if(io[j]>p99_max) p99_max=io[j];
        ++j;
    }

    // Intensity normalization: scale to [0,1] if max > 1.0
    // (approximates p99 normalization without full sort; max is conservative)
    if(p99_max>1.0) {
        double scale=1.0/p99_max;
        for(int i=0;i<M;++i) io[i]*=scale;
    }

    return py::make_tuple(pts_out, int_out);
}


// ── voxel_downsample_with_features: LOAM-inspired source feature selection ────
/**
 * Voxel downsampling with per-voxel covariance eigenvalue analysis.
 * Removes "spherical" voxels (high λ_min/λ_max ratio) which correspond to
 * dynamic object blobs or unstructured clutter that pollute GICP correspondences.
 *
 * Improvement over LOAM's 1D curvature: uses 3D covariance eigenvalues, which
 * works for unorganized point clouds (any LiDAR, any scan pattern).
 *
 * Args:
 *   sphere_thresh:     reject voxel if λ_min/λ_max > thresh (default 0.5)
 *   min_pts_classify:  need ≥ this many points to compute eigenvalues (default 4);
 *                      voxels with fewer points are kept as-is (conservative)
 *
 * Returns: (pts_ds, ints_ds) — filtered (M', 3) and (M',) arrays, M' ≤ M
 */
struct DSFeatureState {
    double mean[3] = {0,0,0};
    double M2[9]   = {0};   // 3×3 Welford accumulator
    double sum_i   = 0;
    int    n       = 0;
};

py::tuple voxel_downsample_with_features(
    py::array_t<double, py::array::c_style|py::array::forcecast> xyz_a,
    py::array_t<double, py::array::c_style|py::array::forcecast> int_a,
    double voxel_size,
    double min_range     = 0.5,
    double max_range     = 80.0,
    double sphere_thresh = 0.5,   // reject if λ_min/λ_max > this
    int    min_pts_classify = 4)  // min points to compute eigenvalues
{
    auto xb=xyz_a.request(), ib=int_a.request();
    const int N=(int)xb.shape[0];
    const double* xyz=(double*)xb.ptr;
    const double* ints=(double*)ib.ptr;
    const double inv=1.0/voxel_size;
    const double r2_min=min_range*min_range, r2_max=max_range*max_range;

    std::unordered_map<DSKey,DSFeatureState,DSKeyHash> voxels;
    voxels.reserve(N/4);

    // Pass 1: range filter + per-voxel Welford mean+covariance
    for(int i=0;i<N;++i) {
        double x=xyz[i*3],y=xyz[i*3+1],z=xyz[i*3+2];
        double r2=x*x+y*y+z*z;
        if(r2<r2_min||r2>r2_max) continue;
        DSKey key{(int32_t)std::floor(x*inv),(int32_t)std::floor(y*inv),(int32_t)std::floor(z*inv)};
        auto& s=voxels[key];
        s.n++;
        double w=1.0/s.n;
        double d0=x-s.mean[0],d1=y-s.mean[1],d2=z-s.mean[2];
        s.mean[0]+=d0*w; s.mean[1]+=d1*w; s.mean[2]+=d2*w;
        double e0=x-s.mean[0],e1=y-s.mean[1],e2=z-s.mean[2];
        s.M2[0]+=d0*e0; s.M2[1]+=d0*e1; s.M2[2]+=d0*e2;
        s.M2[3]+=d1*e0; s.M2[4]+=d1*e1; s.M2[5]+=d1*e2;
        s.M2[6]+=d2*e0; s.M2[7]+=d2*e1; s.M2[8]+=d2*e2;
        s.sum_i+=ints[i];
    }

    // Pass 2: classify and filter
    std::vector<double> out_xyz, out_ints;
    out_xyz.reserve(voxels.size()*3);
    out_ints.reserve(voxels.size());
    double p99_max=0.0;

    for(auto& kv:voxels) {
        const DSFeatureState& s=kv.second;
        double iv=s.sum_i/s.n;

        if(s.n < min_pts_classify) {
            // Too few points to classify reliably → keep (conservative)
            out_xyz.push_back(s.mean[0]); out_xyz.push_back(s.mean[1]); out_xyz.push_back(s.mean[2]);
            out_ints.push_back(iv);
            if(iv>p99_max) p99_max=iv;
            continue;
        }

        // Compute eigenvalues of covariance (Eigen ascending order: ev[0] ≤ ev[1] ≤ ev[2])
        int ns=std::max(s.n-1,1);
        Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> M2m(s.M2);
        Matrix3d cov = M2m*(1.0/ns) + 1e-9*Matrix3d::Identity();
        Eigen::SelfAdjointEigenSolver<Matrix3d> eig(cov, Eigen::EigenvaluesOnly);
        const auto& ev=eig.eigenvalues();
        double lam_max=ev[2], lam_min=ev[0];
        if(lam_max<1e-15) { continue; }  // degenerate (all same point) → drop

        // Sphericity = λ_min / λ_max ∈ [0,1]; 0 = anisotropic (edge/plane), 1 = isotropic (blob)
        double sphericity = lam_min / lam_max;
        if(sphericity > sphere_thresh) continue;  // reject dynamic blob / clutter

        out_xyz.push_back(s.mean[0]); out_xyz.push_back(s.mean[1]); out_xyz.push_back(s.mean[2]);
        out_ints.push_back(iv);
        if(iv>p99_max) p99_max=iv;
    }

    const int M=(int)out_ints.size();
    auto pts_out=py::array_t<double>(std::vector<ssize_t>{M,3});
    auto int_out=py::array_t<double>(std::vector<ssize_t>{M});
    double* po=(double*)pts_out.request().ptr;
    double* io=(double*)int_out.request().ptr;
    std::copy(out_xyz.begin(), out_xyz.end(), po);
    std::copy(out_ints.begin(), out_ints.end(), io);

    if(p99_max>1.0) {
        double scale=1.0/p99_max;
        for(int i=0;i<M;++i) io[i]*=scale;
    }
    return py::make_tuple(pts_out, int_out);
}


// ── voxel_downsample_plane_edge: ring-free plane / edge + optional edge merge ─
// Eigenvalues ascending: λ0 ≤ λ1 ≤ λ2. Sphere reject: λ0/λ2 > sphere_thresh.
// Planarity P=(λ1-λ0)/(λ2+eps), linearity L=(λ2-λ1)/(λ2+eps).
// Plane if P>=p_thr && P>=L; edge if L>=l_thr && L>P.
// If plane count < n_min_plane OR force_include_edges, append edge voxels.
// Optional: drop small voxels (no PCA), score-sort, keep top-K (sparse confident features).
struct ScoredFeature {
    double score;
    double x, y, z, i;
};

py::tuple voxel_downsample_plane_edge(
    py::array_t<double, py::array::c_style|py::array::forcecast> xyz_a,
    py::array_t<double, py::array::c_style|py::array::forcecast> int_a,
    double voxel_size,
    double min_range      = 0.5,
    double max_range      = 80.0,
    double sphere_thresh  = 0.5,
    int    min_pts_classify = 4,
    double planarity_thresh = 0.12,
    double linearity_thresh = 0.12,
    int    n_min_plane    = 200,
    bool   force_include_edges = false,
    bool   drop_small_voxels = true,
    int    max_output_features = 4096,
    double min_feature_score = 0.0)
{
    auto xb=xyz_a.request(), ib=int_a.request();
    const int N=(int)xb.shape[0];
    const double* xyz=(double*)xb.ptr;
    const double* ints=(double*)ib.ptr;
    const double inv=1.0/voxel_size;
    const double r2_min=min_range*min_range, r2_max=max_range*max_range;

    std::unordered_map<DSKey,DSFeatureState,DSKeyHash> voxels;
    voxels.reserve(N/4);

    for(int i=0;i<N;++i) {
        double x=xyz[i*3],y=xyz[i*3+1],z=xyz[i*3+2];
        double r2=x*x+y*y+z*z;
        if(r2<r2_min||r2>r2_max) continue;
        DSKey key{(int32_t)std::floor(x*inv),(int32_t)std::floor(y*inv),(int32_t)std::floor(z*inv)};
        auto& s=voxels[key];
        s.n++;
        double w=1.0/s.n;
        double d0=x-s.mean[0],d1=y-s.mean[1],d2=z-s.mean[2];
        s.mean[0]+=d0*w; s.mean[1]+=d1*w; s.mean[2]+=d2*w;
        double e0=x-s.mean[0],e1=y-s.mean[1],e2=z-s.mean[2];
        s.M2[0]+=d0*e0; s.M2[1]+=d0*e1; s.M2[2]+=d0*e2;
        s.M2[3]+=d1*e0; s.M2[4]+=d1*e1; s.M2[5]+=d1*e2;
        s.M2[6]+=d2*e0; s.M2[7]+=d2*e1; s.M2[8]+=d2*e2;
        s.sum_i+=ints[i];
    }

    std::vector<ScoredFeature> planes;
    std::vector<ScoredFeature> edges;
    planes.reserve(voxels.size());
    edges.reserve(voxels.size());

    for(auto& kv:voxels) {
        const DSFeatureState& s=kv.second;
        double iv=s.sum_i/s.n;
        if(s.n < min_pts_classify) {
            if(!drop_small_voxels) {
                planes.push_back({0.0, s.mean[0], s.mean[1], s.mean[2], iv});
            }
            continue;
        }
        int ns=std::max(s.n-1,1);
        Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> M2m(s.M2);
        Matrix3d cov = M2m*(1.0/ns) + 1e-9*Matrix3d::Identity();
        Eigen::SelfAdjointEigenSolver<Matrix3d> eig(cov, Eigen::EigenvaluesOnly);
        const auto& ev=eig.eigenvalues();
        double l0=ev[0], l1=ev[1], l2=ev[2];
        if(l2<1e-15) continue;
        double sph = l0/l2;
        if(sph > sphere_thresh) continue;
        const double eps = 1e-12;
        double P = (l1-l0)/(l2+eps);
        double L = (l2-l1)/(l2+eps);
        bool is_plane = (P >= planarity_thresh && P >= L);
        bool is_edge  = (L >= linearity_thresh && L > P);
        if(is_plane) {
            planes.push_back({P, s.mean[0], s.mean[1], s.mean[2], iv});
        } else if(is_edge) {
            edges.push_back({L, s.mean[0], s.mean[1], s.mean[2], iv});
        }
    }

    const bool need_edges = force_include_edges || (planes.size() < (size_t)std::max(0, n_min_plane));

    std::vector<ScoredFeature> out;
    out.reserve(planes.size() + (need_edges ? edges.size() : 0));
    for(const auto& p : planes) out.push_back(p);
    if(need_edges) {
        for(const auto& e : edges) out.push_back(e);
    }

    if(min_feature_score > 0.0) {
        out.erase(std::remove_if(out.begin(), out.end(),
            [min_feature_score](const ScoredFeature& f) { return f.score < min_feature_score; }),
            out.end());
    }

    if(max_output_features > 0 && (int)out.size() > max_output_features) {
        std::sort(out.begin(), out.end(), [](const ScoredFeature& a, const ScoredFeature& b) {
            return a.score > b.score;
        });
        out.resize((size_t)max_output_features);
    }

    std::vector<double> out_xyz, out_ints;
    out_xyz.reserve(out.size() * 3);
    out_ints.reserve(out.size());
    double p99_max=0.0;
    for(const auto& f : out) {
        out_xyz.push_back(f.x); out_xyz.push_back(f.y); out_xyz.push_back(f.z);
        out_ints.push_back(f.i);
        if(f.i > p99_max) p99_max = f.i;
    }

    const int M=(int)out_ints.size();
    auto pts_out=py::array_t<double>(std::vector<ssize_t>{M,3});
    auto int_out=py::array_t<double>(std::vector<ssize_t>{M});
    double* po=(double*)pts_out.request().ptr;
    double* io=(double*)int_out.request().ptr;
    std::copy(out_xyz.begin(), out_xyz.end(), po);
    std::copy(out_ints.begin(), out_ints.end(), io);

    if(p99_max>1.0) {
        double scale=1.0/p99_max;
        for(int i=0;i<M;++i) io[i]*=scale;
    }
    return py::make_tuple(pts_out, int_out);
}


// ── pybind11 module ────────────────────────────────────────────────────────────

PYBIND11_MODULE(iv_gicp_map, m) {
    m.doc() = "IV-GICP C++ voxel map v2: O(N) insert, cached KDTree, C++ prefilter.";

    py::class_<VoxelMap>(m, "VoxelMap")
        .def(py::init<double,int,int>(), py::arg("voxel_size"), py::arg("min_points")=3,
             py::arg("max_n_per_voxel")=100)
        .def("insert_frame",        &VoxelMap::insert_frame,
             py::arg("xyz"), py::arg("intensities"), py::arg("frame_id"))
        .def("evict_before",        &VoxelMap::evict_before, py::arg("frame_id"))
        .def("evict_far_from",      &VoxelMap::evict_far_from,
             py::arg("cx"), py::arg("cy"), py::arg("cz"), py::arg("max_dist"))
        .def("evict_incoherent",    &VoxelMap::evict_incoherent,
             py::arg("xyz_world"), py::arg("mahal_sq_threshold")=9.0,
             py::arg("min_inlier_ratio")=0.3, py::arg("min_test_points")=3)
        .def("evict_ghost_raycast", &VoxelMap::evict_ghost_raycast,
             py::arg("sx"), py::arg("sy"), py::arg("sz"),
             py::arg("scan_xyz"), py::arg("ghost_threshold")=0.5,
             py::arg("min_pass_through")=3,
             "Free-space consistency check: evict map voxels whose rays are contradicted "
             "by the current scan. sensor_xyz is the ray origin in world frame.")
        .def("build_target_arrays", &VoxelMap::build_target_arrays,
             py::arg("alpha"), py::arg("source_sigma")=0.0,
             py::arg("k_grad_nbrs")=8, py::arg("count_reg_scale")=2.0,
             py::arg("use_entropy_alpha")=false, py::arg("entropy_scale_c")=0.5,
             py::arg("stable_frames")=0,
             py::arg("max_target_voxels")=0,
             "If max_target_voxels>0, use at most that many voxels for registration/KD (ranked by point count n). "
             "0 = use all valid voxels (default). Map storage is unchanged.")
        .def("query_sigma",         &VoxelMap::query_sigma,
             py::arg("pts_world"), py::arg("max_dist"), py::arg("min_threshold")=0.1)
        .def("get_max_condition_number", &VoxelMap::get_max_condition_number)
        .def("size",   &VoxelMap::size)
        .def("__len__",&VoxelMap::size);

    m.def("downsample_and_filter", &downsample_and_filter,
          py::arg("xyz"), py::arg("intensities"),
          py::arg("voxel_size"), py::arg("min_range")=0.5, py::arg("max_range")=80.0,
          "O(N) hash-based range filter + voxel centroid downsampling.");

    m.def("voxel_downsample_with_features", &voxel_downsample_with_features,
          py::arg("xyz"), py::arg("intensities"),
          py::arg("voxel_size"), py::arg("min_range")=0.5, py::arg("max_range")=80.0,
          py::arg("sphere_thresh")=0.5, py::arg("min_pts_classify")=4,
          "Voxel downsampling with spherical-voxel filtering (LOAM-inspired, 3D covariance eigenvalue). "
          "Removes dynamic object blobs (high sphericity) while keeping edge/planar features.");

    m.def("voxel_downsample_plane_edge", &voxel_downsample_plane_edge,
          py::arg("xyz"), py::arg("intensities"),
          py::arg("voxel_size"), py::arg("min_range")=0.5, py::arg("max_range")=80.0,
          py::arg("sphere_thresh")=0.5, py::arg("min_pts_classify")=4,
          py::arg("planarity_thresh")=0.12, py::arg("linearity_thresh")=0.12,
          py::arg("n_min_plane")=200, py::arg("force_include_edges")=false,
          py::arg("drop_small_voxels")=true,
          py::arg("max_output_features")=4096,
          py::arg("min_feature_score")=0.0,
          "Voxel PCA plane/edge classification (ring-free). Appends edge voxels if plane count < n_min_plane "
          "or force_include_edges. drop_small_voxels: skip voxels with < min_pts_classify (no centroid fallback). "
          "max_output_features: keep top scores (0 = unlimited). min_feature_score: drop weak P/L.");

    m.def("set_num_threads", [](int n) { omp_set_num_threads(n); },
          py::arg("n"),
          "Set OpenMP thread count for parallel regions (e.g. intensity gradients, voxel map).");
    m.def("get_num_threads", []() { return omp_get_max_threads(); },
          "Return current OpenMP max thread count.");
}

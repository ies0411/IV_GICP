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
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

namespace py = pybind11;
using Matrix3d = Eigen::Matrix3d;

// ── Voxel key & hash ───────────────────────────────────────────────────────────

struct VoxelKey {
    int32_t x, y, z;
    bool operator==(const VoxelKey& o) const { return x==o.x && y==o.y && z==o.z; }
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
    VoxelMap(double voxel_size, int min_points)
        : vs_(voxel_size), min_pts_(min_points) {}

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
            s.n++; s.last_frame = frame_id;
            double d0=px-s.mean[0], d1=py-s.mean[1], d2=pz-s.mean[2];
            s.mean[0]+=d0/s.n; s.mean[1]+=d1/s.n; s.mean[2]+=d2/s.n;
            double e0=px-s.mean[0], e1=py-s.mean[1], e2=pz-s.mean[2];
            double dv[3]={d0,d1,d2}, ev[3]={e0,e1,e2};
            for(int r=0;r<3;++r) for(int c=0;c<3;++c) s.M2[r*3+c]+=dv[r]*ev[c];
            double di=ints[i]-s.mean_i;
            s.mean_i+=di/s.n; s.M2_i+=di*(ints[i]-s.mean_i);
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

    int size() const { return (int)voxels_.size(); }

    // ── build_target_arrays: covariances + precisions + gradients ────────────
    /**
     * Builds all target arrays in one C++ pass, AND caches the voxel-center KDTree
     * for reuse by query_sigma() (adaptive correspondence distance update).
     *
     * Returns dict{"means_4d":(V,4), "prec":(V,4,4), "grads":(V,3), "means_3d":(V,3)}
     */
    py::dict build_target_arrays(double alpha, double source_sigma,
                                 int k_grad_nbrs, double count_reg_scale)
    {
        const double eps=1e-6, eps_var=1e-4;
        const double vs2=vs_*vs_, ss2=source_sigma*source_sigma;
        const double cr2=count_reg_scale*count_reg_scale, a2=alpha*alpha;
        const Matrix3d I3 = Matrix3d::Identity();

        // 1. Collect valid voxels
        std::vector<const VoxelState*> valid;
        valid.reserve(voxels_.size());
        for(auto& kv:voxels_) if(kv.second.n>=min_pts_) valid.push_back(&kv.second);
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
            const int ns=std::max(s.n-1,1);
            const double inv=1.0/ns;

            Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> M2m(s.M2);
            Matrix3d Sig = M2m*inv + (eps+ss2+cr2/std::max(s.n,1))*I3;
            Matrix3d Og  = Sig.ldlt().solve(I3);

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

        // 5. Intensity gradients (reuse same tree)
        if(alpha>1e-9 && V>=4 && k_grad_nbrs>=3) {
            const int K=std::min(k_grad_nbrs+1,V);
            #pragma omp parallel for schedule(static)
            for(int i=0;i<V;++i) {
                std::vector<size_t> ni(K); std::vector<double> nd(K);
                nanoflann::KNNResultSet<double> rs(K);
                rs.init(ni.data(),nd.data());
                double qd[3]={m3[i*3],m3[i*3+1],m3[i*3+2]};
                cached_tree_->findNeighbors(rs,qd,nanoflann::SearchParameters());

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

        for(int i=0;i<M;++i) {
            size_t idx; double d2;
            nanoflann::KNNResultSet<double> rs(1);
            rs.init(&idx,&d2);
            double qd[3]={pts[i*3],pts[i*3+1],pts[i*3+2]};
            cached_tree_->findNeighbors(rs,qd,nanoflann::SearchParameters());
            if(d2<=max_d2) inlier_dists.push_back(std::sqrt(d2));
        }

        if(inlier_dists.size()<5) return min_threshold;
        size_t mid=inlier_dists.size()/2;
        std::nth_element(inlier_dists.begin(),inlier_dists.begin()+mid,inlier_dists.end());
        return std::max(inlier_dists[mid], min_threshold);
    }

private:
    double vs_;
    int    min_pts_;
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


// ── pybind11 module ────────────────────────────────────────────────────────────

PYBIND11_MODULE(iv_gicp_map, m) {
    m.doc() = "IV-GICP C++ voxel map v2: O(N) insert, cached KDTree, C++ prefilter.";

    py::class_<VoxelMap>(m, "VoxelMap")
        .def(py::init<double,int>(), py::arg("voxel_size"), py::arg("min_points")=3)
        .def("insert_frame",        &VoxelMap::insert_frame,
             py::arg("xyz"), py::arg("intensities"), py::arg("frame_id"))
        .def("evict_before",        &VoxelMap::evict_before, py::arg("frame_id"))
        .def("evict_far_from",      &VoxelMap::evict_far_from,
             py::arg("cx"), py::arg("cy"), py::arg("cz"), py::arg("max_dist"))
        .def("build_target_arrays", &VoxelMap::build_target_arrays,
             py::arg("alpha"), py::arg("source_sigma")=0.0,
             py::arg("k_grad_nbrs")=8, py::arg("count_reg_scale")=2.0)
        .def("query_sigma",         &VoxelMap::query_sigma,
             py::arg("pts_world"), py::arg("max_dist"), py::arg("min_threshold")=0.1)
        .def("size",   &VoxelMap::size)
        .def("__len__",&VoxelMap::size);

    m.def("downsample_and_filter", &downsample_and_filter,
          py::arg("xyz"), py::arg("intensities"),
          py::arg("voxel_size"), py::arg("min_range")=0.5, py::arg("max_range")=80.0,
          "O(N) hash-based range filter + voxel centroid downsampling.");
}

/**
 * iv_gicp_cpp.cpp
 *
 * pybind11 bindings for nanoflann-based 3D KD-tree.
 * Exposes KDTree3D class with query() matching scipy.spatial.cKDTree interface.
 *
 * Build:
 *   python setup_cpp.py build_ext --inplace
 *
 * Usage (Python):
 *   from iv_gicp.cpp import iv_gicp_cpp
 *   tree = iv_gicp_cpp.KDTree3D(points)       # (N, 3) float64 numpy array
 *   dists, indices = tree.query(queries, k, max_dist)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "nanoflann.hpp"

#include <vector>
#include <limits>
#include <stdexcept>
#include <cmath>

namespace py = pybind11;

// ─── Point cloud adapter for nanoflann ───────────────────────────────────────

struct PointCloud3D {
    std::vector<std::array<double, 3>> pts;

    // nanoflann required interface
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx][dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};

// nanoflann KD-tree type alias
using KDTreeIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud3D>,
    PointCloud3D,
    3  /* dimensionality */
>;


// ─── KDTree3D class ───────────────────────────────────────────────────────────

class KDTree3D {
public:
    /**
     * Build KD-tree from (N, 3) float64 numpy array.
     */
    explicit KDTree3D(py::array_t<double, py::array::c_style | py::array::forcecast> points) {
        auto buf = points.request();
        if (buf.ndim != 2 || buf.shape[1] < 3) {
            throw std::invalid_argument("points must be (N, 3) float64 array");
        }
        const size_t N = buf.shape[0];
        const double* ptr = static_cast<double*>(buf.ptr);

        cloud_.pts.resize(N);
        for (size_t i = 0; i < N; ++i) {
            cloud_.pts[i] = {ptr[i * buf.shape[1]],
                              ptr[i * buf.shape[1] + 1],
                              ptr[i * buf.shape[1] + 2]};
        }

        index_ = std::make_unique<KDTreeIndex>(
            3, cloud_,
            nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf size */)
        );
        index_->buildIndex();
    }

    /**
     * Query for k nearest neighbors.
     *
     * Args:
     *   queries:   (M, 3) float64 numpy array
     *   k:         number of neighbors
     *   max_dist:  maximum search distance (points beyond are returned with
     *              index = n_points and dist = max_dist)
     *
     * Returns:
     *   (dists, indices) as (M, k) float64 and (M, k) int64 numpy arrays.
     */
    py::tuple query(
        py::array_t<double, py::array::c_style | py::array::forcecast> queries,
        int k,
        double max_dist = std::numeric_limits<double>::infinity()
    ) {
        auto buf = queries.request();
        if (buf.ndim != 2 || buf.shape[1] < 3) {
            throw std::invalid_argument("queries must be (M, 3) float64 array");
        }
        const size_t M = buf.shape[0];
        const double* qptr = static_cast<double*>(buf.ptr);
        const size_t N = cloud_.pts.size();

        // Output arrays
        auto dists_out   = py::array_t<double>({(py::ssize_t)M, (py::ssize_t)k});
        auto indices_out = py::array_t<int64_t>({(py::ssize_t)M, (py::ssize_t)k});
        double*  d_ptr = static_cast<double*>(dists_out.request().ptr);
        int64_t* i_ptr = static_cast<int64_t*>(indices_out.request().ptr);

        const double max_dist_sq = (max_dist == std::numeric_limits<double>::infinity())
                                    ? std::numeric_limits<double>::infinity()
                                    : max_dist * max_dist;

        const size_t k_sz = static_cast<size_t>(k);
        std::vector<uint32_t> ret_index(k_sz);
        std::vector<double>   ret_dist(k_sz);

        nanoflann::KNNResultSet<double, uint32_t> result_set(k_sz);

        for (size_t qi = 0; qi < M; ++qi) {
            const double* qp = qptr + qi * buf.shape[1];

            result_set.init(ret_index.data(), ret_dist.data());
            index_->findNeighbors(result_set, qp,
                                  nanoflann::SearchParameters() /* exact search */);

            for (int ki = 0; ki < k; ++ki) {
                double dist_sq = (ki < (int)result_set.size()) ? ret_dist[ki]
                                                                : max_dist_sq;
                uint32_t idx   = (ki < (int)result_set.size()) ? ret_index[ki]
                                                                : (uint32_t)N;

                // Filter by max_dist
                if (max_dist_sq != std::numeric_limits<double>::infinity()
                        && dist_sq > max_dist_sq) {
                    dist_sq = max_dist_sq;
                    idx     = (uint32_t)N;
                }

                d_ptr[qi * k + ki] = std::sqrt(dist_sq);   // return distance, not dist²
                i_ptr[qi * k + ki] = (int64_t)idx;
            }
        }

        return py::make_tuple(dists_out, indices_out);
    }

    /** Number of points in the tree. */
    size_t size() const { return cloud_.pts.size(); }

private:
    PointCloud3D                cloud_;
    std::unique_ptr<KDTreeIndex> index_;
};


// ─── pybind11 module ──────────────────────────────────────────────────────────

PYBIND11_MODULE(iv_gicp_cpp, m) {
    m.doc() = "IV-GICP C++ extension: nanoflann-based KD-tree for fast nearest-neighbor search.";

    py::class_<KDTree3D>(m, "KDTree3D",
        "Fast 3D KD-tree using nanoflann. Drop-in for scipy.spatial.cKDTree.")
        .def(py::init<py::array_t<double>>(),
             py::arg("points"),
             "Build KD-tree from (N, 3) float64 numpy array.")
        .def("query",
             &KDTree3D::query,
             py::arg("queries"),
             py::arg("k") = 1,
             py::arg("max_dist") = std::numeric_limits<double>::infinity(),
             "Query for k nearest neighbors. Returns (dists, indices) as (M,k) arrays.")
        .def("size", &KDTree3D::size,
             "Number of points in the tree.");
}

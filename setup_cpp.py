"""
Build script for IV-GICP C++ extensions.

Builds two modules:
  iv_gicp.cpp.iv_gicp_cpp   — legacy nanoflann KDTree (fast_kdtree.py fallback)
  iv_gicp.cpp.iv_gicp_core  — Eigen-based GN ICP core (main speed path)

Usage:
    python setup_cpp.py build_ext --inplace

Requirements:
    pip install pybind11
    apt install libeigen3-dev
    g++ with C++17 support
"""

from setuptools import setup, Extension
import pybind11
import os

_compile_args = [
    "-std=c++17",
    "-O3",
    "-march=native",
    "-ffast-math",
    "-DNDEBUG",
    "-Wall",
    "-Wextra",
]

_include_dirs = [
    pybind11.get_include(),
    "iv_gicp/cpp",
    "/usr/include/eigen3",
]

ext_kdtree = Extension(
    name="iv_gicp.cpp.iv_gicp_cpp",
    sources=["iv_gicp/cpp/iv_gicp_cpp.cpp"],
    include_dirs=_include_dirs,
    language="c++",
    extra_compile_args=_compile_args,
)

ext_core = Extension(
    name="iv_gicp.cpp.iv_gicp_core",
    sources=["iv_gicp/cpp/iv_gicp_core.cpp"],
    include_dirs=_include_dirs,
    language="c++",
    extra_compile_args=_compile_args + ["-fopenmp"],
    extra_link_args=["-fopenmp"],
)

ext_map = Extension(
    name="iv_gicp.cpp.iv_gicp_map",
    sources=["iv_gicp/cpp/iv_gicp_map.cpp"],
    include_dirs=_include_dirs,
    language="c++",
    extra_compile_args=_compile_args + ["-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(
    name="iv-gicp-cpp",
    version="0.2.0",
    packages=["iv_gicp", "iv_gicp.cpp"],
    ext_modules=[ext_kdtree, ext_core, ext_map],
    zip_safe=False,
)

"""
Build script for IV-GICP C++ extension (nanoflann KDTree).

Usage:
    python setup_cpp.py build_ext --inplace

This compiles iv_gicp/cpp/iv_gicp_cpp.so which is loaded by iv_gicp/fast_kdtree.py.
Falls back to scipy.spatial.cKDTree if not compiled.

Requirements:
    pip install pybind11
    g++ or clang++ with C++17 support
"""

from setuptools import setup, Extension
import pybind11
import sys
import os

ext = Extension(
    name="iv_gicp.cpp.iv_gicp_cpp",
    sources=["iv_gicp/cpp/iv_gicp_cpp.cpp"],
    include_dirs=[
        pybind11.get_include(),
        "iv_gicp/cpp",           # for nanoflann.hpp
    ],
    language="c++",
    extra_compile_args=[
        "-std=c++17",
        "-O3",
        "-march=native",         # enable SIMD (AVX2/SSE4) for distance computations
        "-ffast-math",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
    ],
    extra_link_args=[],
)

setup(
    name="iv-gicp-cpp",
    version="0.1.0",
    packages=["iv_gicp", "iv_gicp.cpp"],
    ext_modules=[ext],
    zip_safe=False,
)

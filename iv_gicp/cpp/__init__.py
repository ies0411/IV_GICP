# iv_gicp/cpp/__init__.py
# C++ extension package. Import iv_gicp_cpp if compiled; otherwise empty.
try:
    from . import iv_gicp_cpp
except ImportError:
    pass

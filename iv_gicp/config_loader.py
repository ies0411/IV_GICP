"""
Load pipeline and dataset config from YAML files under config/.

Usage:
  from iv_gicp.config_loader import get_pipeline_config, get_datasets_config
  pipeline_kw = get_pipeline_config()
  datasets = get_datasets_config()
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


def _config_dir(override: Optional[Path] = None) -> Path:
    if override is not None:
        return Path(override)
    # Default: project root / config (parent of iv_gicp package)
    try:
        from iv_gicp import __file__ as _init_path
        root = Path(_init_path).resolve().parent.parent
    except Exception:
        root = Path.cwd()
    return root / "config"


def _expand_vars(obj: Any) -> Any:
    """Recursively expand environment variables in strings."""
    if isinstance(obj, dict):
        return {k: _expand_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_vars(v) for v in obj]
    if isinstance(obj, str) and "$(" in obj:
        return os.path.expandvars(obj)
    return obj


def load_yaml(path: Path, expand_env: bool = True) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required for config loading. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if expand_env:
        data = _expand_vars(data)
    return data


def get_pipeline_config(config_dir: Optional[Path] = None) -> dict:
    """
    Load pipeline.yaml and return a dict suitable for IVGICPPipeline(**result).
    """
    base = _config_dir(config_dir)
    path = base / "pipeline.yaml"
    if not path.exists():
        return {}
    raw = load_yaml(path)
    # Ensure null -> None for Python; YAML null is already None in PyYAML
    return raw


def get_datasets_config(config_dir: Optional[Path] = None) -> dict:
    """
    Load datasets.yaml and return dict of dataset_id -> { loader, loader_kw, label, alpha, params }.
    loader is a string (e.g. 'load_kitti'); caller must resolve to actual function.
    """
    base = _config_dir(config_dir)
    path = base / "datasets.yaml"
    if not path.exists():
        return {}
    raw = load_yaml(path)
    return raw if isinstance(raw, dict) else {}

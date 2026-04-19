"""Load and merge OmegaConf config fragments for the realtime pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


def _project_root() -> Path:
    """Return the project root (two levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def load_config(realtime_yaml: Optional[Path] = None) -> DictConfig:
    """Load the realtime config by merging profile fragments + runtime overrides.

    Merge order (later overrides earlier):
      1. `configs/profile/base.yaml` — variables, normalization, model defaults
      2. `configs/profile/io_in2d_out6h.yaml` — window indices
      3. `configs/profile/model_gnn_transformer.yaml` — model selector
      4. `configs/realtime.yaml` — runtime knobs (or `realtime_yaml` override)

    Args:
        realtime_yaml: Optional override for the runtime YAML path.

    Returns:
        Merged OmegaConf DictConfig.
    """
    root = _project_root()
    profile_dir = root / "configs" / "profile"
    default_runtime = root / "configs" / "realtime.yaml"
    runtime_path = Path(realtime_yaml) if realtime_yaml else default_runtime

    base = OmegaConf.load(profile_dir / "base.yaml")
    io_cfg = OmegaConf.load(profile_dir / "io_in2d_out6h.yaml")
    model_cfg = OmegaConf.load(profile_dir / "model_gnn_transformer.yaml")
    runtime = OmegaConf.load(runtime_path)

    merged = OmegaConf.merge(base, io_cfg, model_cfg, runtime)
    return merged

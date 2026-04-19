# Vendored from regression-sw/src/networks/ @ 2d89767 on 2026-04-19 — DO NOT EDIT.
# Importing this package registers all model factories with _registry.
from . import gnn  # noqa: F401 — registers "gnn"
from . import transformer  # noqa: F401 — registers "transformer"
from . import tcn  # noqa: F401 — registers "tcn" (imported for gnn.py symbol dep)
from . import patchtst  # noqa: F401 — registers "patchtst" (imported for gnn.py symbol dep)
from ._registry import create_model, list_models  # noqa: F401

__all__ = ["create_model", "list_models"]

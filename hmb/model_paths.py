"""Default locations of the model files that ``install.sh`` downloads (listed in ``install/assets.json``).

Environment variables take precedence: ``HMB_METRABS_MODEL`` (MeTRAbs SavedModel folder or
TensorFlow Hub handle) and ``HMB_SAM3_CHECKPOINT`` (SAM 3.1 checkpoint file).  Without them and
without the local files below, MeTRAbs is downloaded through TensorFlow Hub and SAM 3.1 through
the Hugging Face cache on first use.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
#: MeTRAbs EfficientNetV2-XL SavedModel unpacked by the installer (the model behind METRABS_HANDLE).
METRABS_LOCAL = REPO_ROOT / "outputs/assets/metrabs/metrabs_eff2xl_y4_384px_800k_28ds"
#: TensorFlow Hub handle of the same model, used when no local copy exists.
METRABS_HANDLE = "https://bit.ly/metrabs_xl"
#: SAM 3.1 checkpoint downloaded by ``install.sh --with-sam3``.
SAM3_LOCAL = REPO_ROOT / "outputs/assets/sam3/sam3.1_multiplex.pt"


def local_metrabs_model() -> Optional[Path]:
    """:data:`METRABS_LOCAL` if the installer unpacked it, else ``None``."""
    return METRABS_LOCAL if (METRABS_LOCAL / "saved_model.pb").is_file() else None


def default_metrabs_model(environ: Mapping[str, str] = os.environ) -> Optional[str]:
    """``$HMB_METRABS_MODEL``, else the installer's SavedModel if present, else ``None`` (the loader
    then uses the TensorFlow Hub handle)."""
    value = environ.get("HMB_METRABS_MODEL")
    if value:
        return value
    local = local_metrabs_model()
    return str(local) if local else None

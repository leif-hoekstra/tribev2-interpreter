"""
Loads the canonical neutral baseline vector for contrast computation.

The baseline is a ``(20484,)`` vertex-space array representing the average
brain response to emotionally neutral speech. Subtracting it from raw
TribeV2 predictions removes the generic auditory/TTS component and leaves
the stimulus-specific residual.

The shipped baseline was computed by time-averaging TribeV2 predictions for
a neutral text stimulus. Users can rebuild it with a richer corpus via the
``tribe-pipeline build-baseline`` command.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

_BASELINE_PACKAGE = "tribe_pipeline.baseline.data"

BASELINE_NONE = "none"
BASELINE_CANONICAL = "canonical"


class BaselineLoader:
    """Resolves and loads a baseline vector from various sources."""

    def __init__(self, mode=BASELINE_CANONICAL, path=None):
        """
        Parameters
        ----------
        mode : str
            "canonical" to use the shipped neutral baseline,
            "none" to skip baseline subtraction,
            or a filesystem path string to a custom .npy file.
        path : str or Path, optional
            Used when mode is a filesystem path.
        """
        if mode == BASELINE_CANONICAL:
            self._mode = BASELINE_CANONICAL
            self._path = None
        elif mode == BASELINE_NONE:
            self._mode = BASELINE_NONE
            self._path = None
        else:
            self._mode = "file"
            self._path = Path(mode)

    def load(self):
        """Return ``(baseline_vec, version_str)`` where baseline_vec is ``(20484,)``."""
        if self._mode == BASELINE_NONE:
            return None, "none"
        if self._mode == BASELINE_CANONICAL:
            return _load_packaged()
        arr = np.load(self._path)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        return arr, f"file:{self._path.name}"


def _load_packaged():
    with resources.files(_BASELINE_PACKAGE).joinpath("neutral.npy").open("rb") as f:
        arr = np.load(f)
    try:
        with resources.files(_BASELINE_PACKAGE).joinpath("neutral_meta.json").open("r") as f:
            meta = json.load(f)
        version = f"canonical-v{meta.get('version', '1.0')}"
    except Exception:
        version = "canonical-v1.0"
    return arr, version


def load_canonical_baseline():
    """Convenience wrapper: load the shipped canonical baseline."""
    return BaselineLoader(BASELINE_CANONICAL).load()

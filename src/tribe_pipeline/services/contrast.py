"""
Contrast service.

Removes the generic auditory/TTS component from TribeV2 predictions by
subtracting a pre-computed neutral-speech baseline.  The result is a
stimulus-specific contrast tensor in the same (T, 20484) shape.

Without baseline subtraction, primary auditory cortex dominates every
report because TribeV2 always processes a real audio waveform.  Subtracting
a neutral baseline (predictions for emotionally flat speech) leaves only the
activity that is above or below that baseline for the current stimulus.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ContrastService:
    """Subtract a neutral baseline from raw TribeV2 predictions."""

    def __init__(self, baseline_vec=None, version="none"):
        """
        Parameters
        ----------
        baseline_vec : np.ndarray or None
            Shape ``(20484,)``.  If None, apply() is a no-op (pass-through).
        version : str
            Human-readable description of the baseline origin, stored in the
            Report so outputs are reproducible.
        """
        self._baseline = baseline_vec
        self._version = version

    @classmethod
    def from_loader(cls, loader):
        """Build a ContrastService from a BaselineLoader instance."""
        vec, version = loader.load()
        return cls(baseline_vec=vec, version=version)

    @property
    def version(self):
        return self._version

    def apply(self, predictions):
        """
        Subtract the baseline from ``predictions``.

        Parameters
        ----------
        predictions : np.ndarray
            Shape ``(T, 20484)``.

        Returns
        -------
        np.ndarray
            Shape ``(T, 20484)`` contrast tensor.  Equal to ``predictions``
            when no baseline is loaded.
        """
        if self._baseline is None:
            logger.warning(
                "ContrastService: no baseline loaded -- returning raw predictions. "
                "Auditory cortex will dominate. Use --baseline canonical to fix this."
            )
            return predictions

        if self._baseline.shape[0] != predictions.shape[1]:
            raise ValueError(
                f"Baseline shape {self._baseline.shape} does not match "
                f"prediction vertex count {predictions.shape[1]}"
            )

        return predictions - self._baseline[None, :]

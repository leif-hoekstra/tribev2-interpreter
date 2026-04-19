"""
Affect template service.

Scores the current stimulus along published brain-based affect dimensions by
computing the dot product between the contrast prediction vector and pre-built
``(20484,)`` template weight vectors.  Templates must be placed in
``tribe_pipeline/reference/data/affect_templates/`` via the
``tribe-pipeline build-affect-templates`` command.

If no templates are found, the service emits a warning and returns an empty
list.  The rest of the pipeline continues normally; the LLM is informed via
the ``warnings`` field in the Report.

Templates are expected to be unit-normalised on disk so dot products are
directly comparable across dimensions.  The build script handles normalisation.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

import numpy as np

from tribe_pipeline.schemas import AffectDimension

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = "affect_templates"
_TEMPLATE_PKG = "tribe_pipeline.reference.data"

AFFECT_CATALOG = {
    "negative_affect_pines": "PINES (Chang 2015) — general negative affect",
    "pain_nps": "NPS (Wager 2013) — aversive somatic / pain affect",
    "emotion_fear": "Kragel 2015 discrete-emotion pattern — fear",
    "emotion_sadness": "Kragel 2015 discrete-emotion pattern — sadness",
    "emotion_anger": "Kragel 2015 discrete-emotion pattern — anger",
    "emotion_amusement": "Kragel 2015 discrete-emotion pattern — amusement",
    "emotion_surprise": "Kragel 2015 discrete-emotion pattern — surprise",
    "emotion_contentment": "Kragel 2015 discrete-emotion pattern — contentment",
}


class AffectService:
    """Score contrast predictions against published affect templates."""

    def __init__(self, templates=None):
        """
        Parameters
        ----------
        templates : dict or None
            ``{name: (weight_vec (20484,), source_str)}``.
            If None, attempts to load from package data.
        """
        if templates is not None:
            self._templates = templates
        else:
            self._templates = _load_templates()

    @property
    def available(self):
        return len(self._templates) > 0

    def score(self, contrast_predictions):
        """
        Parameters
        ----------
        contrast_predictions : np.ndarray
            Shape ``(T, 20484)``; time-averaged internally.

        Returns
        -------
        list of AffectDimension
        """
        if not self._templates:
            return []

        activity = contrast_predictions.mean(axis=0)
        results = []
        for name, (weights, source) in self._templates.items():
            norm = float(np.linalg.norm(weights))
            if norm < 1e-12:
                continue
            score = round(float(np.dot(activity, weights) / norm), 4)
            results.append(
                AffectDimension(
                    name=name,
                    score=score,
                    template_source=source,
                )
            )
        results.sort(key=lambda a: abs(a.score), reverse=True)
        return results


def _load_templates():
    """Load all .npy files from the affect_templates package directory."""
    templates = {}
    try:
        template_dir = resources.files(_TEMPLATE_PKG).joinpath(_TEMPLATE_DIR)
        for entry in template_dir.iterdir():
            if not str(entry).endswith(".npy"):
                continue
            name = Path(str(entry)).stem
            with entry.open("rb") as f:
                weights = np.load(f)
            source = AFFECT_CATALOG.get(name, f"template:{name}")
            templates[name] = (weights, source)
    except (FileNotFoundError, TypeError, AttributeError):
        pass

    if not templates:
        logger.info(
            "AffectService: no affect templates found. "
            "Run `tribe-pipeline build-affect-templates` to download them. "
            "Continuing without affect dimensions."
        )
    else:
        logger.info("AffectService: loaded %d affect templates", len(templates))

    return templates

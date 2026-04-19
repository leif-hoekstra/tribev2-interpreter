"""
Top-level pipeline orchestrator.

Stitches together:
  TribeService → ContrastService → ParcellationService
                                 → SubcorticalProxyService
                                 → AffectService
                                 → LLMService

The pipeline is deliberately thin — all domain logic lives in the services —
so it can be driven from a CLI, a notebook, or an HTTP handler.
"""

from __future__ import annotations

import logging

from tribe_pipeline.schemas import FullResult

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end: text -> brain predictions -> contrast -> report -> LLM interpretation."""

    def __init__(self, tribe, contrast, parcellation, subcortical=None, affect=None, llm=None):
        self.tribe = tribe
        self.contrast = contrast
        self.parcellation = parcellation
        self.subcortical = subcortical
        self.affect = affect
        self.llm = llm

    def run(self, text, skip_llm=False):
        """
        Run the full pipeline.

        Parameters
        ----------
        text : str
            The input stimulus.
        skip_llm : bool
            Skip the LLM call (useful for offline or debug runs).

        Returns
        -------
        FullResult
        """
        logger.info("Encoding text with TribeV2 (len=%d chars)", len(text))
        predictions = self.tribe.encode(text)
        logger.info("Predictions shape: %s", predictions.shape)

        logger.info("Applying contrast subtraction (baseline=%s)", self.contrast.version)
        contrast_tensor = self.contrast.apply(predictions)

        logger.info("Building parcellated report")
        report = self.parcellation.build_report(
            contrast_tensor, text, baseline_version=self.contrast.version
        )

        if self.subcortical is not None:
            logger.info("Scoring subcortical proxy regions")
            report.subcortical.extend(self.subcortical.score(contrast_tensor))

        if self.affect is not None and self.affect.available:
            logger.info("Scoring affect dimensions")
            report.affect_dimensions.extend(self.affect.score(contrast_tensor))
        elif self.affect is not None:
            report.warnings.append(
                "Affect templates not found. Run `tribe-pipeline build-affect-templates` to enable."
            )

        emotion_profile = None
        if not skip_llm and self.llm is not None:
            logger.info("Requesting LLM interpretation")
            emotion_profile = self.llm.interpret(report, text)

        return FullResult(
            stimulus_text=text,
            predictions=predictions,
            contrast=contrast_tensor,
            report=report,
            emotion_profile=emotion_profile,
            interpretation=emotion_profile.reasoning if emotion_profile else None,
        )

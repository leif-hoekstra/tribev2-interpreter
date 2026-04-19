"""
LLM interpretation service — v2.

Three-stream reasoning: cortical parcels with per-parcel anatomical labels +
subcortical proxy scores + affect template dimensions.  The LLM emits
structured JSON so the pipeline can programmatically extract valence, arousal,
and dominant emotions alongside free-form prose.
"""

from __future__ import annotations

import json
import logging
import os
from importlib import resources

from tribe_pipeline.config import DEFAULT_LLM_MODEL
from tribe_pipeline.schemas import EmotionProfile

logger = logging.getLogger(__name__)

_PROMPT_PACKAGE = "tribe_pipeline.prompts"


def _load_system_prompt():
    try:
        return (
            resources.files(_PROMPT_PACKAGE)
            .joinpath("interpret_brain.md")
            .read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise RuntimeError(
            "Could not load packaged prompt tribe_pipeline.prompts/interpret_brain.md. "
            "Ensure the package is installed correctly."
        ) from exc


SYSTEM_PROMPT = _load_system_prompt()


class LLMServiceError(RuntimeError):
    """Raised when the LLM call cannot be made (missing deps or key)."""


class LLMService:
    """OpenAI-backed interpreter for brain-activation reports."""

    def __init__(self, model=DEFAULT_LLM_MODEL, api_key=None, temperature=0.3):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise LLMServiceError(
                "OPENAI_API_KEY is not set. Add it to .env or pass api_key= explicitly."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMServiceError(
                "openai package is not installed. Run: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def interpret(self, report, stimulus_text=None):
        """
        Turn a ``Report`` into structured emotion output.

        The stimulus text is intentionally withheld from the LLM so that the
        interpretation is grounded in the neural signal alone, not the text
        content.  This keeps the output honest and testable.

        Returns
        -------
        EmotionProfile
            Structured emotion prediction with prose reasoning.
        """
        client = self._get_client()
        user_message = (
            f"Brain activation report:\n{json.dumps(report.to_dict(), indent=2)}"
        )
        logger.info("Requesting interpretation from model=%s", self._model)
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return _parse_emotion_profile(raw)


def _parse_emotion_profile(raw_json):
    """Parse LLM JSON response into an EmotionProfile; best-effort on malformed output."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON; wrapping as reasoning-only profile")
        return EmotionProfile(
            predicted_valence=0.0,
            predicted_arousal=0.0,
            dominant_emotions=[],
            confidence="low",
            consistency="Unable to parse structured response.",
            reasoning=raw_json,
        )

    return EmotionProfile(
        predicted_valence=float(data.get("predicted_valence", 0.0)),
        predicted_arousal=float(data.get("predicted_arousal", 0.0)),
        dominant_emotions=list(data.get("dominant_emotions", [])),
        confidence=str(data.get("confidence", "low")),
        consistency=str(data.get("consistency", "")),
        reasoning=str(data.get("reasoning", "")),
    )

"""
Runtime configuration for the pipeline.

Settings are resolved in this order:
    1. Explicit argument passed to ``Settings(...)`` or CLI flag
    2. Environment variable (loaded from ``.env`` if present)
    3. Default value
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


TRIBE_CHECKPOINT = "facebook/tribev2"
DEFAULT_CACHE_FOLDER = "./cache"
DEFAULT_OUTPUT_DIR = "./out"
DEFAULT_LLM_MODEL = "gpt-5.4" # Thinking model would work better but for MVP this is fine
DEFAULT_BASELINE = "canonical" # none for no subtractions (raw predictions) -> I have had better results with canonical

CPU_CONFIG_UPDATE = {
    "data.text_feature.device": "cpu",
    "data.audio_feature.device": "cpu",
    "data.video_feature.image.device": "cpu",
    "data.image_feature.image.device": "cpu",
    "accelerator": "cpu",
} 


@dataclass
class Settings:
    tribe_checkpoint: str = TRIBE_CHECKPOINT
    cache_folder: str = DEFAULT_CACHE_FOLDER
    output_dir: str = DEFAULT_OUTPUT_DIR
    llm_model: str = DEFAULT_LLM_MODEL
    baseline: str = DEFAULT_BASELINE
    hf_token: object = None
    openai_api_key: object = None
    config_update: dict = field(default_factory=lambda: dict(CPU_CONFIG_UPDATE))
    skip_llm: bool = False

    @classmethod
    def from_env(cls, **overrides):
        """Build Settings pulling tokens from environment variables."""
        base = dict(
            hf_token=os.environ.get("HF_TOKEN"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)

    def output_path(self):
        return Path(self.output_dir)

"""
TribeV2 encoding service.

The only contract the rest of the pipeline depends on is::

    encode(text: str) -> numpy.ndarray  # shape (T, 20484)

This module ships one implementation that loads the Facebook ``tribev2``
package in-process.  To use a different backend — an HTTP server, a mock
for testing, or any other model — replace this class with one that satisfies
the same ``encode`` signature; nothing else in the pipeline needs to change.

Example HTTP replacement::

    class HttpTribeService:
        def __init__(self, url):
            self._url = url

        def encode(self, text):
            import numpy as np, requests
            resp = requests.post(self._url, json={"text": text}, timeout=120)
            resp.raise_for_status()
            return np.array(resp.json()["predictions"])  # expect (T, 20484)

The in-process implementation below loads the model lazily and caches it on
the instance so repeated calls in the same process reuse the loaded weights.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from tribe_pipeline.config import CPU_CONFIG_UPDATE, TRIBE_CHECKPOINT

logger = logging.getLogger(__name__)


class TribeService:
    """Loads and runs Facebook's TribeV2 brain-encoding model."""

    def __init__(
        self,
        checkpoint=TRIBE_CHECKPOINT,
        cache_folder="./cache",
        config_update=None,
    ):
        self._checkpoint = checkpoint
        self._cache_folder = cache_folder
        self._config_update = dict(config_update) if config_update else dict(CPU_CONFIG_UPDATE)
        self._model = None

    @property
    def is_loaded(self):
        return self._model is not None

    def load(self):
        """Load the TribeV2 model (no-op if already loaded)."""
        if self._model is not None:
            return self._model

        from tribev2 import TribeModel

        Path(self._cache_folder).mkdir(parents=True, exist_ok=True)
        logger.info("Loading TribeV2 model from %s", self._checkpoint)
        self._model = TribeModel.from_pretrained(
            self._checkpoint,
            cache_folder=self._cache_folder,
            config_update=self._config_update,
        )
        return self._model

    def encode(self, text):
        """
        Predict brain responses for a text stimulus.

        Parameters
        ----------
        text : str
            Raw text input. It will be written to a temp file, run through
            TribeV2's TTS + transcription pipeline, and then through the
            trained brain encoder.

        Returns
        -------
        numpy.ndarray
            Shape ``(T, 20484)``: T timesteps x cortical vertices on fsaverage5.
        """
        model = self.load()

        text_path = self._write_temp_text(text)
        try:
            events_df = model.get_events_dataframe(text_path=text_path)
            preds, _segments = model.predict(events=events_df)
        finally:
            os.unlink(text_path)

        preds_np = preds.numpy() if hasattr(preds, "numpy") else np.asarray(preds)
        return preds_np

    @staticmethod
    def _write_temp_text(text):
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        return path

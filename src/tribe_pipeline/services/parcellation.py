"""
Parcellation + network-aggregation service.

Takes a contrast ``(T, 20484)`` tensor (raw predictions minus neutral baseline)
and produces a structured ``Report`` with:
  - all 7 Yeo functional networks ranked by mean z-score
  - top 20 + bottom 10 parcels with per-parcel anatomical annotations
  - temporal dynamics

Pure numpy + reference data. No model, no network calls.
"""

from __future__ import annotations

import numpy as np

from tribe_pipeline.reference import ReferenceData, load_reference_data
from tribe_pipeline.schemas import (
    NetworkStat,
    ParcelStat,
    Report,
    TemporalDynamics,
)

TOP_PARCELS = 20
SUPPRESSED_PARCELS = 10


class ParcellationService:
    """Aggregates vertex-level contrast predictions into named parcels + networks."""

    def __init__(self, reference=None):
        self._ref = reference if reference is not None else load_reference_data()
        self._parcel_ids = self._ref.parcel_ids()
        self._parcel_names = self._ref.parcel_names()

    def parcellate(self, predictions):
        """Average the 20484 vertices into the Destrieux parcels.

        Returns ``(T, n_parcels)`` array.
        """
        if predictions.ndim != 2 or predictions.shape[1] != self._ref.labels_per_vertex.shape[0]:
            raise ValueError(
                f"Expected predictions of shape (T, {self._ref.labels_per_vertex.shape[0]}), "
                f"got {predictions.shape}"
            )
        T = predictions.shape[0]
        parcel_data = np.zeros((T, len(self._parcel_ids)))
        for col, pid in enumerate(self._parcel_ids):
            mask = self._ref.labels_per_vertex == pid
            parcel_data[:, col] = predictions[:, mask].mean(axis=1)
        return parcel_data

    @staticmethod
    def zscore(parcel_data):
        """Z-score each timestep across the parcel axis."""
        mean = parcel_data.mean(axis=1, keepdims=True)
        std = parcel_data.std(axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (parcel_data - mean) / std

    def build_report(self, predictions, stimulus_text, baseline_version="none"):
        """Run the full parcellation + ranking pipeline and return a Report.

        Parameters
        ----------
        predictions : np.ndarray
            Shape ``(T, 20484)``; should be contrast-subtracted for best results.
        stimulus_text : str
            Original text stimulus (used for summary field).
        baseline_version : str
            Version string of the baseline used; stored in Report for provenance.
        """
        warnings = []
        if baseline_version == "none":
            warnings.append(
                "No baseline applied. Auditory cortex will dominate results. "
                "Use --baseline canonical for meaningful emotion-level analysis."
            )

        parcel_data = self.parcellate(predictions)
        zscored_time = self.zscore(parcel_data)
        zscored_mean = zscored_time.mean(axis=0)

        ranked = np.argsort(zscored_mean)[::-1]
        top = self._build_parcel_stats(ranked[:TOP_PARCELS], zscored_mean, zscored_time)
        suppressed = self._build_parcel_stats(
            ranked[-SUPPRESSED_PARCELS:][::-1], zscored_mean, zscored_time, include_terms=False
        )
        all_networks = self._rank_networks(zscored_mean)

        peak_t = int(np.argmax(zscored_time.mean(axis=1)))
        T = zscored_time.shape[0]
        dynamics = TemporalDynamics(
            peak_timestep=peak_t,
            note=f"Mean activation peaks at timestep {peak_t} of {T}.",
        )

        return Report(
            stimulus_summary=_summarize(stimulus_text),
            n_timesteps=T,
            baseline_version=baseline_version,
            all_networks=all_networks,
            top_activated_parcels=top,
            suppressed_parcels=suppressed,
            temporal_dynamics=dynamics,
            warnings=warnings,
        )

    def _build_parcel_stats(self, indices, zscored_mean, zscored_time, include_terms=True):
        out = []
        for i in indices:
            name = self._parcel_names[i]
            annot = self._ref.annotation(name)
            net_key = self._ref.parcel_to_network.get(name, "unknown")
            net_info = self._ref.network_descriptions.get(net_key, {})
            out.append(
                ParcelStat(
                    parcel=name,
                    functional_role=annot.get("functional_role", name),
                    network=net_info.get("name", net_key),
                    z=round(float(zscored_mean[i]), 3),
                    peak_timestep=int(np.argmax(zscored_time[:, i])),
                    terms=list(annot.get("terms", [])) if include_terms else [],
                    affect_relevance=float(annot.get("affect_relevance", 0.0)),
                )
            )
        return out

    def _rank_networks(self, zscored_mean):
        buckets = {}
        for i, name in enumerate(self._parcel_names):
            key = self._ref.parcel_to_network.get(name, "unknown")
            buckets.setdefault(key, []).append(float(zscored_mean[i]))

        result = []
        for key, scores in buckets.items():
            info = self._ref.network_descriptions.get(key, {})
            scores_arr = np.array(scores)
            result.append(
                NetworkStat(
                    network_key=key,
                    name=info.get("name", key),
                    mean_z=round(float(np.mean(scores_arr)), 3),
                    associated_terms=list(info.get("terms", [])),
                    n_parcels=len(scores),
                    n_above_zero=int(np.sum(scores_arr > 0)),
                )
            )
        result.sort(key=lambda n: n.mean_z, reverse=True)
        return result

    @property
    def parcel_names(self):
        return self._parcel_names


def _summarize(text, max_len=120):
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."

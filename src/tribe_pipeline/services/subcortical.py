"""
Subcortical proxy service.

Scores emotionally-relevant subcortical structures that are not directly
predicted by TribeV2 (because they lie outside the fsaverage5 cortical
surface) using weighted sums of functionally-coupled cortical parcels.

Three status tiers:
  direct     — the structure has a cortical surface analogue (e.g. sgACC).
  proxy      — estimated from weighted cortical coupling (e.g. amygdala).
  unavailable — no reliable cortical proxy exists (e.g. hypothalamus, PAG).

Proxy and unavailable entries are explicitly labeled so the LLM is instructed
to present proxy scores as inferences, not measurements, and to say nothing
about unavailable structures.
"""

from __future__ import annotations

import logging

import numpy as np

from tribe_pipeline.schemas import SubcorticalStat

logger = logging.getLogger(__name__)


class SubcorticalProxyService:
    """Score subcortical emotion hubs from contrast predictions."""

    def __init__(self, proxy_table, parcellation):
        """
        Parameters
        ----------
        proxy_table : dict
            Loaded from ``subcortical_proxies.json``.
        parcellation : ParcellationService
            Used to project vertex-level contrast to parcel space.
        """
        self._table = proxy_table
        self._parc = parcellation

    def score(self, contrast_predictions):
        """
        Parameters
        ----------
        contrast_predictions : np.ndarray
            Shape ``(T, 20484)``.

        Returns
        -------
        list of SubcorticalStat
        """
        parcel_data = self._parc.parcellate(contrast_predictions)
        parcel_mean = parcel_data.mean(axis=0)
        name_to_z = dict(zip(self._parc.parcel_names, parcel_mean))

        results = []
        for region, entry in self._table.items():
            status = entry.get("status", "unavailable")

            if status == "direct":
                direct_parcel = entry.get("direct_parcel", "")
                z = name_to_z.get(direct_parcel, 0.0)
                results.append(
                    SubcorticalStat(
                        region=region,
                        status="direct",
                        score=round(float(z), 3),
                        confidence=entry.get("confidence", "high"),
                        limitations=entry.get("limitations", ""),
                    )
                )

            elif status == "proxy":
                proxies = entry.get("proxies", [])
                score = sum(
                    p["weight"] * name_to_z.get(p["parcel"], 0.0)
                    for p in proxies
                )
                contributors = [p["parcel"] for p in proxies]
                results.append(
                    SubcorticalStat(
                        region=region,
                        status="proxy",
                        score=round(float(score), 3),
                        confidence=entry.get("confidence", "moderate"),
                        contributors=contributors,
                        limitations=entry.get("limitations", ""),
                    )
                )

            else:
                results.append(
                    SubcorticalStat(
                        region=region,
                        status="unavailable",
                        limitations=entry.get("limitations", "Not on cortical surface."),
                    )
                )

        return results

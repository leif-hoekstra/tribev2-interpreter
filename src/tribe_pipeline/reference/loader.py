"""
Loads all shipped reference data for brain parcellation and annotation.

Reference files live in ``tribe_pipeline/reference/data/`` and are packaged
with the wheel via ``package-data`` in pyproject.toml.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

import numpy as np

_DATA_PACKAGE = "tribe_pipeline.reference.data"


@dataclass(frozen=True)
class ReferenceData:
    """Immutable bundle of parcellation + network lookups + per-parcel annotations."""

    labels_per_vertex: np.ndarray
    parcel_index_to_name: dict
    parcel_to_network: dict
    network_descriptions: dict
    parcel_annotations: dict
    subcortical_proxies: dict

    def parcel_ids(self):
        """Return sorted parcel IDs, excluding the background (0)."""
        return sorted(set(self.labels_per_vertex.tolist()) - {0})

    def parcel_names(self):
        """Return parcel names in the same order as ``parcel_ids()``."""
        return [self.parcel_index_to_name[str(pid)] for pid in self.parcel_ids()]

    def annotation(self, parcel_name):
        """Return the parcel annotation dict, falling back to network-level terms."""
        if parcel_name in self.parcel_annotations:
            return self.parcel_annotations[parcel_name]
        net_key = self.parcel_to_network.get(parcel_name, "unknown")
        net_info = self.network_descriptions.get(net_key, {})
        return {
            "functional_role": parcel_name,
            "terms": net_info.get("terms", []),
            "affect_relevance": 0.0,
            "notes": "",
        }


def _read_json(filename):
    with resources.files(_DATA_PACKAGE).joinpath(filename).open("r") as f:
        return json.load(f)


def _read_npy(filename):
    with resources.files(_DATA_PACKAGE).joinpath(filename).open("rb") as f:
        return np.load(f)


def load_reference_data():
    """Load all reference data bundled with the package."""
    return ReferenceData(
        labels_per_vertex=_read_npy("destrieux_labels.npy"),
        parcel_index_to_name=_read_json("destrieux_parcels.json"),
        parcel_to_network=_read_json("parcel_to_network.json"),
        network_descriptions=_read_json("network_descriptions.json"),
        parcel_annotations=_read_json("parcel_annotations.json"),
        subcortical_proxies=_read_json("subcortical_proxies.json"),
    )

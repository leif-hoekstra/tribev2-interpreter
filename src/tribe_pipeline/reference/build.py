"""
Rebuilds the shipped reference data from upstream sources.

This is a one-time developer script. End users do not need to run it --
the reference files are already packaged with the wheel.

Run from the project root:
    python -m tribe_pipeline.reference.build
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from nilearn import datasets


PARCEL_TO_NETWORK = {
    "G_and_S_frontomargin": "frontoparietal",
    "G_and_S_occipital_inf": "visual",
    "G_and_S_paracentral": "somatomotor",
    "G_and_S_subcentral": "somatomotor",
    "G_and_S_transv_frontopol": "frontoparietal",
    "G_and_S_cingul-Ant": "default",
    "G_and_S_cingul-Mid-Ant": "default",
    "G_and_S_cingul-Mid-Post": "default",
    "G_cingul-Post-dorsal": "default",
    "G_cingul-Post-ventral": "default",
    "G_cuneus": "visual",
    "G_front_inf-Opercular": "frontoparietal",
    "G_front_inf-Orbital": "frontoparietal",
    "G_front_inf-Triangul": "frontoparietal",
    "G_front_middle": "frontoparietal",
    "G_front_sup": "default",
    "G_Ins_lg_and_S_cent_ins": "ventral_attention",
    "G_insular_short": "ventral_attention",
    "G_occipital_middle": "visual",
    "G_occipital_sup": "visual",
    "G_oc-temp_lat-fusifor": "visual",
    "G_oc-temp_med-Lingual": "visual",
    "G_oc-temp_med-Parahip": "default",
    "G_orbital": "default",
    "G_pariet_inf-Angular": "default",
    "G_pariet_inf-Supramar": "dorsal_attention",
    "G_parietal_sup": "dorsal_attention",
    "G_postcentral": "somatomotor",
    "G_precentral": "somatomotor",
    "G_precuneus": "default",
    "G_rectus": "default",
    "G_subcallosal": "limbic",
    "G_temp_sup-G_T_transv": "somatomotor",
    "G_temp_sup-G_temp_transv_and_interm_S": "somatomotor",
    "G_temp_sup-Lateral": "default",
    "G_temp_sup-Plan_polar": "somatomotor",
    "G_temp_sup-Plan_tempo": "ventral_attention",
    "G_temporal_inf": "default",
    "G_temporal_middle": "default",
    "Lat_Fis-ant-Horizont": "ventral_attention",
    "Lat_Fis-ant-Vertical": "ventral_attention",
    "Lat_Fis-post": "somatomotor",
    "Medial_wall": "default",
    "Pole_occipital": "visual",
    "Pole_temporal": "default",
    "S_calcarine": "visual",
    "S_central": "somatomotor",
    "S_cingul-Marginalis": "default",
    "S_circular_insula_ant": "ventral_attention",
    "S_circular_insula_inf": "ventral_attention",
    "S_circular_insula_sup": "somatomotor",
    "S_collat_transv_ant": "default",
    "S_collat_transv_post": "visual",
    "S_front_inf": "frontoparietal",
    "S_front_middle": "frontoparietal",
    "S_front_sup": "frontoparietal",
    "S_interm_prim-Jensen": "somatomotor",
    "S_intrapariet_and_P_trans": "dorsal_attention",
    "S_oc_middle_and_Lunatus": "visual",
    "S_oc_sup_and_transversal": "visual",
    "S_occipital_ant": "visual",
    "S_oc-temp_lat": "visual",
    "S_oc-temp_med_and_Lingual": "visual",
    "S_orbital_lateral": "default",
    "S_orbital_med-olfact": "limbic",
    "S_orbital-H_Shaped": "default",
    "S_parieto_occipital": "dorsal_attention",
    "S_pericallosal": "default",
    "S_postcentral": "somatomotor",
    "S_precentral-inf-part": "somatomotor",
    "S_precentral-sup-part": "somatomotor",
    "S_suborbital": "limbic",
    "S_subparietal": "default",
    "S_temporal_inf": "default",
    "S_temporal_sup": "default",
    "S_temporal_transverse": "somatomotor",
}

NETWORK_DESCRIPTIONS = {
    "visual": {
        "name": "Visual Network",
        "terms": [
            "visual perception",
            "object recognition",
            "early visual processing",
            "spatial representation",
        ],
    },
    "somatomotor": {
        "name": "Somatomotor Network",
        "terms": [
            "motor control",
            "sensory processing",
            "body representation",
            "auditory processing",
        ],
    },
    "dorsal_attention": {
        "name": "Dorsal Attention Network",
        "terms": [
            "top-down attention",
            "spatial attention",
            "goal-directed processing",
            "working memory",
        ],
    },
    "ventral_attention": {
        "name": "Ventral Attention Network",
        "terms": [
            "stimulus-driven attention",
            "salience",
            "interoception",
            "language articulation",
        ],
    },
    "limbic": {
        "name": "Limbic Network",
        "terms": [
            "emotion regulation",
            "reward processing",
            "olfaction",
            "motivation",
        ],
    },
    "frontoparietal": {
        "name": "Frontoparietal Network",
        "terms": [
            "cognitive control",
            "executive function",
            "working memory",
            "flexible task control",
            "language production",
        ],
    },
    "default": {
        "name": "Default Mode Network",
        "terms": [
            "self-referential thought",
            "autobiographical memory",
            "social cognition",
            "mind-wandering",
            "semantic processing",
            "narrative comprehension",
        ],
    },
}


def build(output_dir=None):
    """Fetch the Destrieux atlas and write all reference files."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching Destrieux surface atlas from nilearn...")
    atlas = datasets.fetch_atlas_surf_destrieux()

    labels = [str(lbl) for lbl in atlas["labels"]]
    combined = np.concatenate([np.array(atlas["map_left"]), np.array(atlas["map_right"])])
    if combined.shape != (20484,):
        raise RuntimeError(f"Expected (20484,) parcellation, got {combined.shape}")

    np.save(output_dir / "destrieux_labels.npy", combined)
    with open(output_dir / "destrieux_parcels.json", "w") as f:
        json.dump({str(i): n for i, n in enumerate(labels)}, f, indent=2)
    with open(output_dir / "parcel_to_network.json", "w") as f:
        json.dump(PARCEL_TO_NETWORK, f, indent=2)
    with open(output_dir / "network_descriptions.json", "w") as f:
        json.dump(NETWORK_DESCRIPTIONS, f, indent=2)

    atlas_named = set(labels) - {"Unknown"}
    missing = atlas_named - set(PARCEL_TO_NETWORK.keys())
    if missing:
        raise RuntimeError(f"Parcels without network assignment: {sorted(missing)}")

    print(f"Wrote {len(labels)} parcels and {len(NETWORK_DESCRIPTIONS)} networks to {output_dir}")


if __name__ == "__main__":
    build()

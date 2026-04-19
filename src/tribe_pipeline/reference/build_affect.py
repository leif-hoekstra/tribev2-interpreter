"""
Build affect templates for fsaverage5 from publicly available NIfTI maps.

This is a one-time developer script.  End users run it via:
    tribe-pipeline build-affect-templates

Sources:
  PINES  — Chang et al. 2015, PLoS Biology
           (NeuroVault collection 306, image "Rating_Weights_LOSO_2")
  Kragel — Kragel & LaBar 2015, SCAN
           (NeuroVault collection 12383, 3-component PLS group maps)

NPS (Wager et al. 2013) is intentionally NOT fetched here because the
canonical weight map is only distributed through the CANlab lab under a
usage agreement (Masks_Private/NPS_share) and has no public direct-download
URL.  If you have obtained it, resample to fsaverage5, unit-normalise, and
drop it into ``reference/data/affect_templates/pain_nps.npy`` — AffectService
will pick it up automatically.

Requirements:
  pip install nilearn  (already a dependency)
  pip install requests  (for downloading)
  Internet access
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "data" / "affect_templates"

_KRAGEL_BASE = "https://neurovault.org/media/images/12383"

NEUROVAULT_IMAGES = {
    "negative_affect_pines": {
        "url": "https://neurovault.org/media/images/306/Rating_Weights_LOSO_2.nii.gz",
        "description": "PINES — general negative affect (Chang 2015)",
    },
    "emotion_fear": {
        "url": f"{_KRAGEL_BASE}/mean_3comp_fearful_group_emotion_PLS_beta_BSz_10000it.nii.gz",
        "description": "Kragel 2015 — discrete emotion: fear",
    },
    "emotion_sadness": {
        "url": f"{_KRAGEL_BASE}/mean_3comp_sad_group_emotion_PLS_beta_BSz_10000it.nii.gz",
        "description": "Kragel 2015 — discrete emotion: sadness",
    },
    "emotion_anger": {
        "url": f"{_KRAGEL_BASE}/mean_3comp_angry_group_emotion_PLS_beta_BSz_10000it.nii.gz",
        "description": "Kragel 2015 — discrete emotion: anger",
    },
    "emotion_amusement": {
        "url": f"{_KRAGEL_BASE}/mean_3comp_amused_group_emotion_PLS_beta_BSz_10000it.nii.gz",
        "description": "Kragel 2015 — discrete emotion: amusement",
    },
    "emotion_contentment": {
        "url": f"{_KRAGEL_BASE}/mean_3comp_content_group_emotion_PLS_beta_BSz_10000it.nii.gz",
        "description": "Kragel 2015 — discrete emotion: contentment",
    },
    "emotion_surprise": {
        "url": f"{_KRAGEL_BASE}/mean_3comp_surprised_group_emotion_PLS_beta_BSz_10000it.nii.gz",
        "description": "Kragel 2015 — discrete emotion: surprise",
    },
}


def build_all(output_dir=None):
    """Download all templates and resample to fsaverage5."""
    if output_dir is None:
        output_dir = _TEMPLATE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import requests
        from nilearn import image, surface, datasets
    except ImportError as exc:
        raise RuntimeError(
            "nilearn and requests are required. "
            "Run: pip install nilearn requests"
        ) from exc

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    manifest = {}
    success = 0

    for name, info in NEUROVAULT_IMAGES.items():
        out_path = output_dir / f"{name}.npy"
        if out_path.exists():
            logger.info("  %s already exists, skipping", name)
            manifest[name] = {"source": info["url"]}
            success += 1
            continue

        print(f"Downloading {name} from NeuroVault...")
        try:
            resp = requests.get(info["url"], timeout=120)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  FAILED: {exc}. Skipping.")
            continue

        tmp_path = output_dir / f"_tmp_{name}.nii.gz"
        tmp_path.write_bytes(resp.content)

        try:
            nii = image.load_img(str(tmp_path))
            surf_l = surface.vol_to_surf(nii, fsaverage["pial_left"])
            surf_r = surface.vol_to_surf(nii, fsaverage["pial_right"])
            combined = np.concatenate([surf_l, surf_r]).astype(np.float32)
            nan_mask = np.isnan(combined)
            combined[nan_mask] = 0.0
            norm = np.linalg.norm(combined)
            if norm > 1e-12:
                combined /= norm
            np.save(out_path, combined)
            manifest[name] = {"source": info["url"]}
            print(f"  Saved {name} ({combined.shape}, norm={norm:.3f})")
            success += 1
        except Exception as exc:
            print(f"  ERROR resampling {name}: {exc}")
        finally:
            tmp_path.unlink(missing_ok=True)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBuilt {success}/{len(NEUROVAULT_IMAGES)} templates → {output_dir}")
    if success < len(NEUROVAULT_IMAGES):
        print("Some templates failed. Re-run to retry, or source them manually.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_all()

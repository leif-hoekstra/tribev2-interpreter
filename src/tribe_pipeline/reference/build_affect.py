"""
Build affect templates for fsaverage5 from publicly available NIfTI maps.

This is a one-time developer script.  End users run it via:
    tribe-pipeline build-affect-templates

Sources:
  PINES  — Chang et al. 2015, NeuroVault collection 503
  NPS    — Wager et al. 2013, NeuroVault collection 504
  Kragel — Kragel et al. 2021, NeuroVault collection 8245

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

NEUROVAULT_IMAGES = {
    "negative_affect_pines": {
        "url": "https://neurovault.org/media/images/503/pines_final.nii.gz",
        "description": "PINES — general negative affect (Chang 2015)",
    },
    "pain_nps": {
        "url": "https://neurovault.org/media/images/504/weights_NSF_grouppred_cvpcr.img",
        "description": "NPS — pain / aversive somatic (Wager 2013)",
    },
    "emotion_fear": {
        "url": "https://neurovault.org/media/images/8245/Fear_SVM.nii.gz",
        "description": "Kragel 2021 — discrete emotion: fear",
    },
    "emotion_sadness": {
        "url": "https://neurovault.org/media/images/8245/Sadness_SVM.nii.gz",
        "description": "Kragel 2021 — discrete emotion: sadness",
    },
    "emotion_anger": {
        "url": "https://neurovault.org/media/images/8245/Anger_SVM.nii.gz",
        "description": "Kragel 2021 — discrete emotion: anger",
    },
    "emotion_joy": {
        "url": "https://neurovault.org/media/images/8245/Joy_SVM.nii.gz",
        "description": "Kragel 2021 — discrete emotion: joy",
    },
    "emotion_amusement": {
        "url": "https://neurovault.org/media/images/8245/Amusement_SVM.nii.gz",
        "description": "Kragel 2021 — discrete emotion: amusement",
    },
    "emotion_contentment": {
        "url": "https://neurovault.org/media/images/8245/Contentment_SVM.nii.gz",
        "description": "Kragel 2021 — discrete emotion: contentment",
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
            manifest[name] = {"path": str(out_path), "source": info["url"]}
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
            manifest[name] = {"path": str(out_path), "source": info["url"]}
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

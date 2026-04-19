"""
Affect validation harness.

Runs the TribeV2 pipeline on labeled emotional stimulus sets and checks
whether affect dimensions discriminate between emotion categories.

Pass criteria (v1, lenient):
  - sad > happy on emotion_sadness (Cohen's d > 0.3)
  - fearful > neutral on PINES negative_affect (d > 0.3)
  - neutral near zero across all affect dimensions

Usage:
    tribe-pipeline validate
    python -m tribe_pipeline.validation.validate_affect

IMPORTANT: Requires TribeV2 model and a working environment. Slow (processes
~60 sentences). Use for offline benchmarking, not CI.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_STIMULUS_DIR = Path(__file__).parent / "stimuli"
CATEGORIES = ["happy", "sad", "fearful", "neutral"]


def _cohens_d(a, b):
    """Cohen's d effect size between two 1-D arrays."""
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def run(cache_folder="./cache", baseline="canonical", output_dir=None):
    """Run the validation harness and print a report."""
    from dotenv import load_dotenv
    load_dotenv()

    from tribe_pipeline.baseline.loader import BaselineLoader
    from tribe_pipeline.config import Settings
    from tribe_pipeline.reference.loader import load_reference_data
    from tribe_pipeline.services import (
        AffectService,
        ContrastService,
        ParcellationService,
        SubcorticalProxyService,
        TribeService,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ref = load_reference_data()
    tribe = TribeService(cache_folder=cache_folder)
    contrast_svc = ContrastService.from_loader(BaselineLoader(baseline))
    parc = ParcellationService(reference=ref)
    sub = SubcorticalProxyService(ref.subcortical_proxies, parc)
    affect = AffectService()

    if not affect.available:
        print(
            "ERROR: No affect templates loaded. "
            "Run `tribe-pipeline build-affect-templates` first."
        )
        return

    results = {}

    for category in CATEGORIES:
        txt_path = _STIMULUS_DIR / f"{category}.txt"
        sentences = [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]
        print(f"\nProcessing {category} ({len(sentences)} sentences)...")

        cat_scores = {}
        for i, sent in enumerate(sentences, start=1):
            print(f"  [{i}/{len(sentences)}] {sent[:50]}...")
            preds = tribe.encode(sent)
            contrast = contrast_svc.apply(preds)
            dims = affect.score(contrast)
            for d in dims:
                cat_scores.setdefault(d.name, []).append(d.score)

        results[category] = {name: np.array(scores) for name, scores in cat_scores.items()}

    print("\n" + "=" * 60)
    print("AFFECT DIMENSION MEANS BY CATEGORY")
    print("=" * 60)
    all_dims = sorted({dim for cat in results.values() for dim in cat})
    header = f"{'dim':<30}" + "".join(f"{cat:>12}" for cat in CATEGORIES)
    print(header)
    for dim in all_dims:
        row = f"{dim:<30}" + "".join(
            f"{np.mean(results[cat].get(dim, np.array([0]))):.3f}".rjust(12)
            for cat in CATEGORIES
        )
        print(row)

    print("\n" + "=" * 60)
    print("KEY CONTRASTS (Cohen's d)")
    print("=" * 60)

    def _d(cat_a, cat_b, dim):
        a = results[cat_a].get(dim, np.array([0.0]))
        b = results[cat_b].get(dim, np.array([0.0]))
        return _cohens_d(a, b)

    checks = [
        ("sad > happy on emotion_sadness", "sad", "happy", "emotion_sadness", 0.3),
        ("fearful > neutral on negative_affect_pines", "fearful", "neutral", "negative_affect_pines", 0.3),
        ("fearful > neutral on emotion_fear", "fearful", "neutral", "emotion_fear", 0.3),
        ("happy > sad on emotion_joy", "happy", "sad", "emotion_joy", 0.3),
    ]

    pass_count = 0
    for label, cat_a, cat_b, dim, threshold in checks:
        d = _d(cat_a, cat_b, dim)
        passed = d >= threshold
        status = "PASS" if passed else "FAIL"
        if passed:
            pass_count += 1
        print(f"  [{status}] {label}: d={d:.3f} (threshold={threshold})")

    print(f"\n{pass_count}/{len(checks)} checks passed.")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary = {
            cat: {dim: float(np.mean(scores)) for dim, scores in cat_data.items()}
            for cat, cat_data in results.items()
        }
        (out / "validation_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nSaved summary to {out / 'validation_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run affect validation harness.")
    parser.add_argument("--cache-folder", default="./cache")
    parser.add_argument("--baseline", default="canonical")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(cache_folder=args.cache_folder, baseline=args.baseline, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

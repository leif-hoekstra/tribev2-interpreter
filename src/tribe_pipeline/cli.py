"""
Command-line interface for tribe-pipeline.

Subcommands:
  run                 Run the full pipeline end-to-end.
  compare             Compare two stimuli (A vs B) via differential analysis.
  build-baseline      Build a new canonical neutral baseline from a text corpus.
  build-affect-templates  Download and resample published affect templates.
  validate            Run the affect validation harness.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from tribe_pipeline import __version__
from tribe_pipeline.baseline.loader import BaselineLoader
from tribe_pipeline.config import Settings
from tribe_pipeline.io import write_result
from tribe_pipeline.pipeline import Pipeline
from tribe_pipeline.services import (
    AffectService,
    ContrastService,
    LLMService,
    ParcellationService,
    SubcorticalProxyService,
    TribeService,
)


def _configure_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _read_text(args):
    if getattr(args, "text", None) and getattr(args, "text_file", None):
        raise SystemExit("error: pass either --text or --text-file, not both")
    if getattr(args, "text", None):
        return args.text
    if getattr(args, "text_file", None):
        return Path(args.text_file).read_text()
    raise SystemExit("error: one of --text or --text-file is required")


def _build_pipeline(settings):
    """Instantiate and wire all services."""
    from tribe_pipeline.reference.loader import load_reference_data

    ref = load_reference_data()

    tribe = TribeService(
        cache_folder=settings.cache_folder,
        config_update=settings.config_update,
    )

    baseline_loader = BaselineLoader(settings.baseline)
    contrast = ContrastService.from_loader(baseline_loader)

    parcellation = ParcellationService(reference=ref)
    subcortical = SubcorticalProxyService(ref.subcortical_proxies, parcellation)
    affect = AffectService()

    llm = None
    if not settings.skip_llm:
        llm = LLMService(model=settings.llm_model, api_key=settings.openai_api_key)

    return Pipeline(
        tribe=tribe,
        contrast=contrast,
        parcellation=parcellation,
        subcortical=subcortical,
        affect=affect,
        llm=llm,
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="tribe-pipeline",
        description="TribeV2 brain encoding + emotion interpretation pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"tribe-pipeline {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- run ----
    run = sub.add_parser("run", help="Run the full pipeline end-to-end.")
    run.add_argument("--text", help="Inline stimulus text.")
    run.add_argument("--text-file", help="Path to a text file containing the stimulus.")
    run.add_argument(
        "--output-dir", default="./out",
        help="Output directory for .npy, .json, .md (default: ./out)",
    )
    run.add_argument(
        "--cache-folder", default="./cache",
        help="TribeV2 model cache folder (default: ./cache)",
    )
    run.add_argument("--model", default=None, help="LLM model (default: gpt-5.4).")
    run.add_argument(
        "--baseline", default="canonical",
        help="Baseline for contrast: 'canonical', 'none', or path to .npy (default: canonical)",
    )
    run.add_argument("--skip-llm", action="store_true", help="Skip the LLM interpretation step.")
    run.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")

    # ---- compare ----
    cmp = sub.add_parser("compare", help="Compare two stimuli A and B.")
    cmp.add_argument("--text-a", required=True, help="Stimulus A text.")
    cmp.add_argument("--text-b", required=True, help="Stimulus B text.")
    cmp.add_argument("--output-dir", default="./out")
    cmp.add_argument("--cache-folder", default="./cache")
    cmp.add_argument("--model", default=None)
    cmp.add_argument("--baseline", default="canonical")
    cmp.add_argument("--skip-llm", action="store_true")
    cmp.add_argument("-v", "--verbose", action="store_true")

    # ---- build-baseline ----
    bb = sub.add_parser(
        "build-baseline",
        help="Build a canonical neutral baseline from a text corpus.",
    )
    bb.add_argument(
        "--corpus", required=True,
        help="Path to a text file with one sentence per line.",
    )
    bb.add_argument(
        "--output", default="./baseline.npy",
        help="Path to write the output .npy baseline (default: ./baseline.npy)",
    )
    bb.add_argument("--cache-folder", default="./cache")
    bb.add_argument("-v", "--verbose", action="store_true")

    # ---- build-affect-templates ----
    bat = sub.add_parser(
        "build-affect-templates",
        help="Download and resample published affect templates (requires nilearn, internet).",
    )
    bat.add_argument("-v", "--verbose", action="store_true")

    # ---- validate ----
    val = sub.add_parser(
        "validate",
        help="Run the affect validation harness on labeled stimulus sets.",
    )
    val.add_argument("--cache-folder", default="./cache")
    val.add_argument("--baseline", default="canonical")
    val.add_argument("-v", "--verbose", action="store_true")

    return parser


# ---------- subcommand handlers ----------

def run_command(args):
    load_dotenv()
    settings = Settings.from_env(
        cache_folder=args.cache_folder,
        output_dir=args.output_dir,
        llm_model=args.model,
        skip_llm=args.skip_llm,
        baseline=args.baseline,
    )
    text = _read_text(args)
    pipeline = _build_pipeline(settings)
    result = pipeline.run(text, skip_llm=settings.skip_llm)
    paths = write_result(result, settings.output_path())
    print("Wrote:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    return 0


def compare_command(args):
    load_dotenv()
    settings = Settings.from_env(
        cache_folder=args.cache_folder,
        output_dir=args.output_dir,
        llm_model=args.model,
        skip_llm=args.skip_llm,
        baseline=args.baseline,
    )
    pipeline = _build_pipeline(settings)
    result_a = pipeline.run(args.text_a, skip_llm=settings.skip_llm)
    result_b = pipeline.run(args.text_b, skip_llm=settings.skip_llm)

    from tribe_pipeline.io.writers import write_comparison
    out_dir = settings.output_path()
    paths = write_comparison(result_a, result_b, out_dir)
    print("Wrote comparison:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    return 0


def build_baseline_command(args):
    load_dotenv()
    import json as _json

    import numpy as np

    corpus_path = Path(args.corpus)
    sentences = [ln.strip() for ln in corpus_path.read_text().splitlines() if ln.strip()]
    if not sentences:
        raise SystemExit(f"error: no sentences found in {corpus_path}")

    output_path = Path(args.output)
    cache_folder = args.cache_folder

    print(f"Building baseline from {len(sentences)} sentences...")
    tribe = TribeService(cache_folder=cache_folder)

    accum = None
    n = 0
    for i, sent in enumerate(sentences, start=1):
        print(f"  [{i}/{len(sentences)}] {sent[:60]}")
        preds = tribe.encode(sent)
        collapsed = preds.mean(axis=0)
        if accum is None:
            accum = collapsed.copy()
        else:
            accum += collapsed
        n += 1

    baseline = accum / n
    np.save(output_path, baseline)

    meta = {
        "n_sentences": n,
        "corpus_file": str(corpus_path),
        "version": "1.0",
    }
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)

    print(f"Wrote baseline ({baseline.shape}) to {output_path}")
    print(f"Wrote metadata to {meta_path}")
    return 0


def build_affect_templates_command(args):
    """Download and resample PINES, NPS, Kragel 2021 templates."""
    from tribe_pipeline.reference.build_affect import build_all
    build_all()
    return 0


def validate_command(args):
    load_dotenv()
    _configure_logging(getattr(args, "verbose", False))
    try:
        from tribe_pipeline.validation.validate_affect import run as _run
    except ImportError:
        raise SystemExit(
            "error: validation module not found. "
            "Install the package in editable mode: pip install -e ."
        )
    _run(cache_folder=args.cache_folder, baseline=args.baseline)
    return 0


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))

    dispatch = {
        "run": run_command,
        "compare": compare_command,
        "build-baseline": build_baseline_command,
        "build-affect-templates": build_affect_templates_command,
        "validate": validate_command,
    }

    handler = dispatch.get(args.command)
    if handler:
        return handler(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

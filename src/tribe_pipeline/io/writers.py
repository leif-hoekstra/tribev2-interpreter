"""
Output writers.

Serialize a ``FullResult`` to disk as:
  - predictions.npy    raw (T, 20484) predictions
  - contrast.npy       contrast tensor (predictions - baseline)
  - report.json        structured data + emotion profile
  - interpretation.md  human-readable summary with emotion dashboard
  - emotion_profile.json  structured LLM output (if LLM ran)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def write_result(result, output_dir):
    """Write all artifacts for a ``FullResult`` into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    preds_path = output_dir / "predictions.npy"
    np.save(preds_path, np.asarray(result.predictions))

    contrast_path = output_dir / "contrast.npy"
    if result.contrast is not None:
        np.save(contrast_path, np.asarray(result.contrast))

    json_payload = {
        "timestamp": timestamp,
        "stimulus_text": result.stimulus_text,
        "predictions_path": preds_path.name,
        "predictions_shape": list(np.asarray(result.predictions).shape),
        "baseline_version": result.report.baseline_version,
        "report": result.report.to_dict(),
        "emotion_profile": result.emotion_profile.to_dict() if result.emotion_profile else None,
        "llm_reasoning": result.interpretation,
    }
    json_path = output_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    emotion_path = None
    if result.emotion_profile is not None:
        emotion_path = output_dir / "emotion_profile.json"
        with open(emotion_path, "w") as f:
            json.dump(result.emotion_profile.to_dict(), f, indent=2)

    md_path = output_dir / "interpretation.md"
    with open(md_path, "w") as f:
        f.write(_render_markdown(result, timestamp))

    out = {"predictions": preds_path, "report": json_path, "interpretation": md_path}
    if emotion_path:
        out["emotion_profile"] = emotion_path
    return out


def write_comparison(result_a, result_b, output_dir):
    """Write a differential comparison of two FullResult objects."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    md_path = output_dir / "comparison.md"
    json_path = output_dir / "comparison.json"

    md = _render_comparison_markdown(result_a, result_b, timestamp)
    with open(md_path, "w") as f:
        f.write(md)

    diff_payload = _build_comparison_dict(result_a, result_b, timestamp)
    with open(json_path, "w") as f:
        json.dump(diff_payload, f, indent=2)

    return {"comparison_md": md_path, "comparison_json": json_path}


# ---------- markdown renderers ----------

def _render_markdown(result, timestamp):
    report = result.report
    ep = result.emotion_profile
    lines = [
        "# TribeV2 Brain Interpretation",
        "",
        f"**Generated:** {timestamp}",
        f"**Baseline:** {report.baseline_version}",
        "",
        "## Stimulus",
        "",
        f"> {result.stimulus_text}",
        "",
    ]

    if ep:
        lines += [
            "## Emotion Profile",
            "",
            f"| Dimension | Value |",
            f"|---|---|",
            f"| Predicted valence | {ep.predicted_valence:+.2f} |",
            f"| Predicted arousal | {ep.predicted_arousal:.2f} |",
            f"| Dominant emotions | {', '.join(ep.dominant_emotions) if ep.dominant_emotions else 'none'} |",
            f"| Confidence | {ep.confidence} |",
            f"| Stream consistency | {ep.consistency} |",
            "",
        ]

    if report.affect_dimensions:
        lines += ["## Affect Dimensions", ""]
        sorted_af = sorted(report.affect_dimensions, key=lambda a: abs(a.score), reverse=True)
        for a in sorted_af:
            bar = _bar(a.score, -0.5, 0.5)
            lines.append(f"  {a.name}: {a.score:+.3f}  {bar}")
        lines.append("")

    lines += ["## All Networks (ranked)", ""]
    for i, n in enumerate(report.all_networks, start=1):
        terms = ", ".join(n.associated_terms[:3])
        active = f"{n.n_above_zero}/{n.n_parcels} parcels above zero"
        lines.append(f"  {i}. **{n.name}** mean z={n.mean_z:+.3f} | {active} | {terms}")
    lines.append("")

    if report.subcortical:
        lines += ["## Subcortical Inference", ""]
        for s in report.subcortical:
            if s.status == "unavailable":
                continue
            tag = "[DIRECT]" if s.status == "direct" else "[PROXY]"
            lines.append(f"  {tag} **{s.region}**: z={s.score:+.3f} ({s.confidence})")
        lines.append("")

    lines += ["## Top Activated Parcels (z-contrast)", ""]
    for i, p in enumerate(report.top_activated_parcels, start=1):
        terms = ", ".join(p.terms[:4])
        ar = f"affect={p.affect_relevance:.1f}" if p.affect_relevance > 0 else ""
        lines.append(
            f"  {i}. **{p.parcel}** — {p.functional_role}"
        )
        lines.append(
            f"     Network: {p.network} | z={p.z:+.3f} | peak t={p.peak_timestep} | {ar}"
        )
        if terms:
            lines.append(f"     Terms: {terms}")
    lines.append("")

    lines += ["## Suppressed Parcels", ""]
    for p in report.suppressed_parcels:
        lines.append(
            f"  - **{p.parcel}** ({p.functional_role}, {p.network}, z={p.z:+.3f})"
        )
    lines.append("")

    td = report.temporal_dynamics
    lines += ["## Temporal Dynamics", "", f"- {td.note}", ""]

    if report.warnings:
        lines += ["## Warnings", ""]
        for w in report.warnings:
            lines.append(f"  - {w}")
        lines.append("")

    if ep:
        lines += ["## LLM Interpretation", "", ep.reasoning, ""]
    elif result.interpretation:
        lines += ["## LLM Interpretation", "", result.interpretation, ""]
    else:
        lines += ["## LLM Interpretation", "", "_Skipped._", ""]

    return "\n".join(lines)


def _render_comparison_markdown(result_a, result_b, timestamp):
    lines = [
        "# Stimulus Comparison",
        "",
        f"**Generated:** {timestamp}",
        "",
        f"**Stimulus A:** {result_a.stimulus_text}",
        f"**Stimulus B:** {result_b.stimulus_text}",
        "",
        "## Affect Dimension Deltas (A - B)",
        "",
    ]

    af_a = {a.name: a.score for a in result_a.report.affect_dimensions}
    af_b = {a.name: a.score for a in result_b.report.affect_dimensions}
    all_dims = sorted(set(af_a) | set(af_b))
    for dim in all_dims:
        delta = af_a.get(dim, 0) - af_b.get(dim, 0)
        lines.append(f"  {dim}: {delta:+.3f}")
    lines.append("")

    lines += ["## Network Deltas (A mean_z - B mean_z)", ""]
    net_a = {n.name: n.mean_z for n in result_a.report.all_networks}
    net_b = {n.name: n.mean_z for n in result_b.report.all_networks}
    for name in net_a:
        delta = net_a.get(name, 0) - net_b.get(name, 0)
        lines.append(f"  {name}: {delta:+.3f}")
    lines.append("")

    lines += ["## Subcortical Deltas (A score - B score)", ""]
    sub_a = {s.region: s.score for s in result_a.report.subcortical if s.status != "unavailable"}
    sub_b = {s.region: s.score for s in result_b.report.subcortical if s.status != "unavailable"}
    for region in sub_a:
        delta = sub_a.get(region, 0) - sub_b.get(region, 0)
        lines.append(f"  {region}: {delta:+.3f}")
    lines.append("")

    if result_a.emotion_profile and result_b.emotion_profile:
        ep_a, ep_b = result_a.emotion_profile, result_b.emotion_profile
        lines += [
            "## Emotion Profile Comparison",
            "",
            f"| | A | B |",
            f"|---|---|---|",
            f"| Valence | {ep_a.predicted_valence:+.2f} | {ep_b.predicted_valence:+.2f} |",
            f"| Arousal | {ep_a.predicted_arousal:.2f} | {ep_b.predicted_arousal:.2f} |",
            f"| Dominant | {', '.join(ep_a.dominant_emotions)} | {', '.join(ep_b.dominant_emotions)} |",
            f"| Confidence | {ep_a.confidence} | {ep_b.confidence} |",
            "",
        ]

    return "\n".join(lines)


def _build_comparison_dict(result_a, result_b, timestamp):
    return {
        "timestamp": timestamp,
        "stimulus_a": result_a.stimulus_text,
        "stimulus_b": result_b.stimulus_text,
        "report_a": result_a.report.to_dict(),
        "report_b": result_b.report.to_dict(),
        "emotion_profile_a": result_a.emotion_profile.to_dict() if result_a.emotion_profile else None,
        "emotion_profile_b": result_b.emotion_profile.to_dict() if result_b.emotion_profile else None,
    }


def _bar(value, lo, hi, width=10):
    """Tiny ASCII bar chart for affect score display."""
    ratio = (value - lo) / (hi - lo)
    ratio = max(0, min(1, ratio))
    filled = int(round(ratio * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"

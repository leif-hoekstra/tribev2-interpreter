"""
Data schemas for brain-encoding reports.

All records are plain dataclasses with a ``to_dict`` helper so they can be
serialized to JSON without pulling in pydantic or similar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ParcelStat:
    parcel: str
    network: str
    z: float
    functional_role: str = ""
    peak_timestep: int = 0
    terms: list = field(default_factory=list)
    affect_relevance: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class NetworkStat:
    network_key: str
    name: str
    mean_z: float
    associated_terms: list = field(default_factory=list)
    n_parcels: int = 0
    n_above_zero: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class SubcorticalStat:
    region: str
    status: str
    score: float = 0.0
    confidence: str = "n/a"
    contributors: list = field(default_factory=list)
    limitations: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class AffectDimension:
    name: str
    score: float
    template_source: str

    def to_dict(self):
        return asdict(self)


@dataclass
class EmotionProfile:
    predicted_valence: float
    predicted_arousal: float
    dominant_emotions: list
    confidence: str
    consistency: str
    reasoning: str

    def to_dict(self):
        return asdict(self)


@dataclass
class TemporalDynamics:
    peak_timestep: int
    note: str

    def to_dict(self):
        return asdict(self)


@dataclass
class Report:
    """Structured summary of a brain-encoding prediction, ready for LLM interpretation."""

    stimulus_summary: str
    n_timesteps: int
    baseline_version: str
    all_networks: list
    top_activated_parcels: list
    suppressed_parcels: list
    temporal_dynamics: TemporalDynamics
    subcortical: list = field(default_factory=list)
    affect_dimensions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "stimulus_summary": self.stimulus_summary,
            "n_timesteps": self.n_timesteps,
            "baseline_version": self.baseline_version,
            "all_networks": [n.to_dict() for n in self.all_networks],
            "top_activated_parcels": [p.to_dict() for p in self.top_activated_parcels],
            "suppressed_parcels": [p.to_dict() for p in self.suppressed_parcels],
            "subcortical": [s.to_dict() for s in self.subcortical],
            "affect_dimensions": [a.to_dict() for a in self.affect_dimensions],
            "temporal_dynamics": self.temporal_dynamics.to_dict(),
            "warnings": self.warnings,
        }


@dataclass
class FullResult:
    """End-to-end pipeline output: raw predictions + report + optional LLM prose."""

    stimulus_text: str
    predictions: object
    contrast: object
    report: Report
    emotion_profile: object = None
    interpretation: object = None

    def report_dict(self):
        return self.report.to_dict()

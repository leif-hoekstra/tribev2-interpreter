"""Brain encoding + emotion interpretation pipeline built on Facebook's TribeV2."""

from tribe_pipeline.config import Settings
from tribe_pipeline.pipeline import Pipeline
from tribe_pipeline.schemas import (
    AffectDimension,
    EmotionProfile,
    FullResult,
    NetworkStat,
    ParcelStat,
    Report,
    SubcorticalStat,
)
from tribe_pipeline.services import (
    AffectService,
    ContrastService,
    LLMService,
    ParcellationService,
    SubcorticalProxyService,
    TribeService,
)

__version__ = "0.2.0"

__all__ = [
    "Pipeline",
    "Settings",
    "TribeService",
    "ContrastService",
    "ParcellationService",
    "SubcorticalProxyService",
    "AffectService",
    "LLMService",
    "Report",
    "FullResult",
    "ParcelStat",
    "NetworkStat",
    "SubcorticalStat",
    "AffectDimension",
    "EmotionProfile",
    "__version__",
]

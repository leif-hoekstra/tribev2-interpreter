<div align="center">

# TribeV2 Interpreter

**Text → Brain Predictions → Emotion Interpretation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model: facebook/tribev2](https://img.shields.io/badge/%F0%9F%A4%97%20Model-facebook%2Ftribev2-orange)](https://huggingface.co/facebook/tribev2)

</div>

TribeV2 Interpreter is a brain-encoding + emotion-interpretation pipeline built on top of Meta's [TribeV2](https://huggingface.co/facebook/tribev2) foundation model. It takes a piece of natural language text, runs it through a multimodal brain encoder to predict the cortical response an average human subject would produce when listening to that text read aloud, and then distils the raw 20 484-vertex prediction into named brain regions, subcortical proxy scores, and published affect signatures — before handing all three to a large language model that reasons about what the neural pattern reveals about the emotional content of the stimulus.

---

## How it works

```
Text input
    │
    ▼
TribeV2 (facebook/tribev2)
    Converts text → speech → word-level events,
    runs them through a trained brain encoder,
    returns a (T × 20 484) cortical prediction
    on the fsaverage5 surface.
    │
    ▼
Contrast subtraction
    Removes the "neutral speech" baseline so
    only the stimulus-specific signal remains.
    │
    ▼
Three evidence streams
    ├── Parcellation  — collapses the 20 484 vertices into
    │                   labelled Destrieux parcels and Yeo 7-networks,
    │                   each annotated with cognitive/affective priors.
    ├── Subcortical   — proxy scores for amygdala, hippocampus,
    │                   ventral striatum, and other deep structures.
    └── Affect dims   — dot-product scores against published
                        brain-based affect signatures
                        (PINES — Chang 2015; Kragel 2015 discrete emotions).
    │
    ▼
LLM interpretation
    All three streams are serialised into a structured JSON
    report and passed to an OpenAI model with a neuroscience
    system prompt. The model returns valence, arousal,
    dominant emotions, and free-form reasoning grounded
    entirely in the neural signal — not the input text.
    │
    ▼
Outputs
    predictions.npy      raw cortical predictions
    report.json          full structured report
    emotion_profile.json valence / arousal / dominant emotions
    interpretation.md    prose reasoning
```

---

## Prerequisites

### 1. TribeV2 (required, separate install)

TribeV2 is **not on PyPI**. You must obtain and install it separately before using this pipeline. Follow the official setup instructions on the [TribeV2 model card](https://huggingface.co/facebook/tribev2) or install from the Meta research repository:

```bash
pip install git+https://github.com/facebookresearch/tribev2
```

Model weight are downloaded automatically from Hugging Face on first use and cached locally.

### 2. OpenAI API key

LLM interpretation requires an OpenAI API key:

```bash
cp .env.example .env
# add your key:  OPENAI_API_KEY=sk-...
```

---

## Installation

```bash
git clone https://github.com/leif-hoekstra/tribev2-interpreter
cd tribev2-interpreter
pip install -e .
```

---

## Quick start

**Run the full pipeline on a text stimulus:**

```bash
tribe-pipeline run \
    --text "She opened the letter and her hands began to tremble." \
    --output-dir ./out
```

**Compare two stimuli:**

```bash
tribe-pipeline compare \
    --text-a "The crowd erupted in applause." \
    --text-b "The room fell completely silent."
```

**Skip the LLM (offline / fast mode):**

```bash
tribe-pipeline run --text "The room fell completely silent." --skip-llm
```

---

## Affect templates

Seven published brain-based affect signatures (PINES — Chang 2015; six Kragel 2015 discrete-emotion classifiers) ship pre-built with this repository as unit-normalised cortical weight vectors. No extra download step is needed.

If you ever need to regenerate them from source (e.g. after updating the NeuroVault URLs or resampling parameters):

```bash
tribe-pipeline build-affect-templates
```

If templates are absent for any reason the pipeline still runs; the LLM is simply told affect dimensions are unavailable.

---

## Output

After a run, the output directory contains:

| File | Contents |
|---|---|
| `report.json` | Full structured report: parcels, networks, subcortical, affect dimensions, warnings |
| `emotion_profile.json` | Structured emotion output: predicted valence, arousal, dominant emotions, confidence |
| `interpretation.md` | Free-form LLM reasoning grounded in the neural signal |
| `predictions.npy` | Raw `(T, 20484)` cortical predictions for downstream analysis |

---

## Architecture overview

The pipeline is deliberately thin — all domain logic lives in swappable services:

| Service | Responsibility |
|---|---|
| `TribeService` | Wraps TribeV2; the only required interface is `encode(text) → ndarray`. Swap in an HTTP backend or a mock without touching anything else. |
| `ContrastService` | Subtracts the neutral baseline; supports `canonical`, `none`, or a custom `.npy`. |
| `ParcellationService` | Maps vertices → Destrieux parcels → Yeo networks; annotates each with affective priors. |
| `SubcorticalProxyService` | Scores deep-structure proxy regions. |
| `AffectService` | Dot-product scoring against affect templates; degrades gracefully if templates are missing. |
| `LLMService` | Calls OpenAI with the three-stream report; returns a structured `EmotionProfile`. |

---

## Third-party data and models

| Source | Used for | Citation |
|---|---|---|
| Meta `facebook/tribev2` | Brain encoding model | [d'Ascoli et al. 2026](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) |
| Destrieux atlas (via nilearn) | Cortical parcellation | Destrieux et al. 2010 |
| Yeo 7-network parcellation | Network aggregation | Yeo et al. 2011 |
| PINES (NeuroVault #306) | Affect template | Chang et al. 2015, *PLoS Biology* |
| Kragel 2015 emotion maps (NeuroVault #12383) | Affect templates | Kragel & LaBar 2015, *SCAN* |

---

## License

This project is licensed under the [MIT License](LICENSE).

TribeV2 model weights are subject to Meta's separate license — see the [Hugging Face model card](https://huggingface.co/facebook/tribev2) for terms. Third-party affect templates downloaded via `build-affect-templates` are subject to their respective NeuroVault licenses.

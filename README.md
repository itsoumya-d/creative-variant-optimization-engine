# Creative Variant Optimization Engine

Production-grade AI optimization engine for evaluating dynamic creative variants (headlines, body copy, CTA), multi-armed bandit (Bayesian Thompson sampling) traffic allocation, and creative fatigue scoring.

## Features
- **Dynamic Creative Scoring**: Evaluates multi-modal creative components across length, sentiment indicators, and conversion affinity.
- **Bayesian Thompson Sampling**: Allocates impression traffic across creative variants balancing exploration and exploitation to maximize click-through rate (CTR).
- **Fatigue & Decay Detection**: Quantifies creative performance degradation over cumulative impression exposures.
- **Strict Validation**: Typed Pydantic models for creative metadata and impression telemetry.

## Testing & Verification
```bash
python3 -m pip install -e ".[test]"
pytest -v
```

import math
import random
from typing import Dict, List, Optional
from .models import CreativeVariant, ImpressionFeedback, AllocationResult


class CreativeOptimizer:
    """Bayesian Thompson Sampling optimizer for dynamic ad creative variants."""

    def __init__(self, rng_seed: Optional[int] = None):
        self._variants: Dict[str, CreativeVariant] = {}
        self._campaign_variants: Dict[str, List[str]] = {}
        self._rng = random.Random(rng_seed)

    def register_variant(self, variant: CreativeVariant) -> None:
        """Register a new creative variant under its campaign."""
        self._variants[variant.variant_id] = variant
        c_list = self._campaign_variants.setdefault(variant.campaign_id, [])
        if variant.variant_id not in c_list:
            c_list.append(variant.variant_id)

    def get_variant(self, variant_id: str) -> Optional[CreativeVariant]:
        return self._variants.get(variant_id)

    def sample_variant(self, campaign_id: str) -> AllocationResult:
        """Select creative variant using Bayesian Thompson Sampling."""
        variant_ids = self._campaign_variants.get(campaign_id, [])
        if not variant_ids:
            raise ValueError(f"No variants registered for campaign: {campaign_id}")

        best_id = ""
        highest_sample = -1.0
        for vid in variant_ids:
            var = self._variants[vid]
            # Draw sample from Beta(alpha, beta) distribution
            sample = self._rng.betavariate(var.alpha, var.beta_param)
            if sample > highest_sample:
                highest_sample = sample
                best_id = vid

        chosen = self._variants[best_id]
        return AllocationResult(
            variant_id=chosen.variant_id,
            campaign_id=chosen.campaign_id,
            sampled_score=round(highest_sample, 6),
            posterior_mean=round(chosen.posterior_mean, 6),
        )

    def record_feedback(self, feedback: ImpressionFeedback) -> None:
        """Update Bayesian posterior and impression counters."""
        var = self._variants.get(feedback.variant_id)
        if not var:
            raise KeyError(f"Variant not found: {feedback.variant_id}")

        var.impressions += 1
        if feedback.converted:
            var.conversions += 1
            var.alpha += 1.0
        else:
            var.beta_param += 1.0

    def calculate_fatigue(self, variant_id: str, half_life_impressions: int = 5000) -> float:
        """Calculate exponential fatigue decay factor in [0.0, 1.0]."""
        var = self._variants.get(variant_id)
        if not var or var.impressions == 0:
            return 1.0
        decay_constant = math.log(2) / half_life_impressions
        fatigue = math.exp(-decay_constant * var.impressions)
        return round(max(0.0, min(1.0, fatigue)), 4)

    def get_campaign_summary(self, campaign_id: str) -> Dict[str, dict]:
        """Return performance telemetry for all variants in a campaign."""
        variant_ids = self._campaign_variants.get(campaign_id, [])
        total_impressions = sum(self._variants[vid].impressions for vid in variant_ids)
        summary = {}
        for vid in variant_ids:
            var = self._variants[vid]
            share = (var.impressions / total_impressions) if total_impressions > 0 else 0.0
            summary[vid] = {
                "impressions": var.impressions,
                "conversions": var.conversions,
                "empirical_ctr": round(var.empirical_ctr, 4),
                "posterior_mean": round(var.posterior_mean, 4),
                "traffic_share": round(share, 4),
                "fatigue_factor": self.calculate_fatigue(vid),
            }
        return summary

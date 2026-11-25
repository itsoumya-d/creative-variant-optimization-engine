from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class CreativeComponent(BaseModel):
    """Component of an ad creative variant (e.g. headline, copy, CTA)."""
    kind: Literal["headline", "body", "cta", "image", "video"]
    text: Optional[str] = None
    asset_url: Optional[str] = None


class CreativeVariant(BaseModel):
    """Creative variant tracked for dynamic optimization."""
    variant_id: str
    campaign_id: str
    components: List[CreativeComponent]
    alpha: float = Field(default=1.0, ge=1.0, description="Beta prior successes (+1)")
    beta_param: float = Field(default=1.0, ge=1.0, description="Beta prior failures (+1)")
    impressions: int = Field(default=0, ge=0)
    conversions: int = Field(default=0, ge=0)

    @property
    def empirical_ctr(self) -> float:
        if self.impressions == 0:
            return 0.0
        return self.conversions / self.impressions

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_param)


class ImpressionFeedback(BaseModel):
    """Impression outcome telemetry."""
    variant_id: str
    campaign_id: str
    converted: bool
    latency_ms: float = 0.0


class AllocationResult(BaseModel):
    """Decision output for variant serving."""
    variant_id: str
    campaign_id: str
    sampled_score: float
    posterior_mean: float

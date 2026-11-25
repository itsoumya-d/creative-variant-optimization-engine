import pytest
from creative_engine.models import CreativeVariant, CreativeComponent, ImpressionFeedback
from creative_engine.optimizer import CreativeOptimizer


@pytest.fixture
def optimizer():
    return CreativeOptimizer(rng_seed=42)


def test_variant_registration_and_properties(optimizer):
    components = [
        CreativeComponent(kind="headline", text="Transform your ads with AI"),
        CreativeComponent(kind="cta", text="Get Started"),
    ]
    variant = CreativeVariant(
        variant_id="var-1",
        campaign_id="camp-100",
        components=components,
    )
    optimizer.register_variant(variant)

    retrieved = optimizer.get_variant("var-1")
    assert retrieved is not None
    assert retrieved.campaign_id == "camp-100"
    assert retrieved.empirical_ctr == 0.0
    assert retrieved.posterior_mean == 0.5  # 1.0 / (1.0 + 1.0)


def test_sample_variant_requires_registration(optimizer):
    with pytest.raises(ValueError, match="No variants registered"):
        optimizer.sample_variant("unknown-campaign")


def test_bayesian_feedback_update(optimizer):
    variant = CreativeVariant(
        variant_id="v-feedback",
        campaign_id="camp-1",
        components=[CreativeComponent(kind="headline", text="Headline A")],
    )
    optimizer.register_variant(variant)

    # Record 3 conversions, 1 failure
    for _ in range(3):
        optimizer.record_feedback(ImpressionFeedback(variant_id="v-feedback", campaign_id="camp-1", converted=True))
    optimizer.record_feedback(ImpressionFeedback(variant_id="v-feedback", campaign_id="camp-1", converted=False))

    updated = optimizer.get_variant("v-feedback")
    assert updated.impressions == 4
    assert updated.conversions == 3
    assert updated.empirical_ctr == 0.75
    assert updated.alpha == 4.0  # 1.0 initial + 3
    assert updated.beta_param == 2.0  # 1.0 initial + 1
    assert round(updated.posterior_mean, 3) == 0.667


def test_thompson_sampling_convergence(optimizer):
    """High CTR variant should receive the clear majority of traffic over time."""
    v_high = CreativeVariant(
        variant_id="var-high",
        campaign_id="camp-ai",
        components=[CreativeComponent(kind="headline", text="Winning Creative")],
    )
    v_low = CreativeVariant(
        variant_id="var-low",
        campaign_id="camp-ai",
        components=[CreativeComponent(kind="headline", text="Losing Creative")],
    )
    optimizer.register_variant(v_high)
    optimizer.register_variant(v_low)

    # Simulate 300 impression iterations with true CTRs: high=70%, low=10%
    true_ctrs = {"var-high": 0.70, "var-low": 0.10}
    selections = {"var-high": 0, "var-low": 0}

    sim_rng = optimizer._rng
    for _ in range(300):
        decision = optimizer.sample_variant("camp-ai")
        vid = decision.variant_id
        selections[vid] += 1

        converted = sim_rng.random() < true_ctrs[vid]
        optimizer.record_feedback(
            ImpressionFeedback(variant_id=vid, campaign_id="camp-ai", converted=converted)
        )

    # Winning variant must have gathered substantially more impressions
    assert selections["var-high"] > selections["var-low"] * 2
    assert selections["var-high"] > 200

    summary = optimizer.get_campaign_summary("camp-ai")
    assert summary["var-high"]["traffic_share"] > 0.65
    assert summary["var-high"]["posterior_mean"] > summary["var-low"]["posterior_mean"]


def test_fatigue_decay_calculation(optimizer):
    variant = CreativeVariant(
        variant_id="var-fatigue",
        campaign_id="camp-decay",
        components=[CreativeComponent(kind="headline", text="Ad")],
        impressions=5000,
    )
    optimizer.register_variant(variant)

    fatigue = optimizer.calculate_fatigue("var-fatigue", half_life_impressions=5000)
    assert abs(fatigue - 0.5) < 0.01  # At half-life, decay should be approx 0.5

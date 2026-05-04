"""Tests for audio-reactive intensity curve contract."""

import pytest
from pydantic import ValidationError

from app.schemas import SetReactiveConfig, UpdateBlockConfig
from hambajuba2ba.bridge.destinations import apply_intensity_curve


@pytest.mark.parametrize("model", [UpdateBlockConfig, SetReactiveConfig])
def test_impulse_is_not_a_supported_intensity_curve(model):
    with pytest.raises(ValidationError):
        model.model_validate(
            {
                "block": "down.2.1",
                "space": "prompt",
                "intensity_curve": "impulse",
            }
        )


def test_intensity_curves_are_stateless_shaping_functions():
    assert apply_intensity_curve(0.5, "linear", 2.0) == 0.5
    assert apply_intensity_curve(0.5, "gamma", 2.0) == 0.25
    assert apply_intensity_curve(0.5, "clip", 2.0) == 0.75

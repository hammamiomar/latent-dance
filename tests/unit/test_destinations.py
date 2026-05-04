"""Tests for destination and composition slot lifecycle."""

import torch

from hambajuba2ba.bridge.composition import CompositionEngine
from hambajuba2ba.bridge.destinations import Destination, DestinationModulator


def _prompt_destination(label: str, value: float) -> Destination:
    return Destination(
        tensor=torch.full((1, 4, 8), value, dtype=torch.float32),
        tensor_pooled=torch.full((1, 4), value, dtype=torch.float32),
        label=label,
        prompt=label,
    )


def test_prompt_clear_a_promotes_b_to_a():
    modulator = DestinationModulator(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    modulator.load_a(_prompt_destination("A", 1.0))
    modulator.load_b(_prompt_destination("B", 2.0))
    modulator.set_blend(0.75)

    modulator.clear_destination("a")

    assert modulator.destination_a is not None
    assert modulator.destination_a.label == "B"
    assert modulator.destination_b is None
    assert modulator.blend_position == 0.0


def test_prompt_clear_b_keeps_a():
    modulator = DestinationModulator(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    modulator.load_a(_prompt_destination("A", 1.0))
    modulator.load_b(_prompt_destination("B", 2.0))
    modulator.set_blend(0.75)

    modulator.clear_destination("b")

    assert modulator.destination_a is not None
    assert modulator.destination_a.label == "A"
    assert modulator.destination_b is None
    assert modulator.blend_position == 0.0


def test_composition_clear_a_promotes_b_to_a():
    comp = CompositionEngine(
        shape=(1, 1, 2, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    comp.load_noise("a", torch.ones((1, 1, 2, 2)), seed=11)
    comp.load_noise("b", torch.full((1, 1, 2, 2), 2.0), seed=22)

    comp.clear_noise("a")

    assert comp.seed_a == 22
    assert comp.seed_b is None
    assert not comp.has_both()
    assert torch.equal(comp.step(0.0), torch.full((1, 1, 2, 2), 2.0))


def test_composition_clear_b_keeps_a():
    comp = CompositionEngine(
        shape=(1, 1, 2, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    comp.load_noise("a", torch.ones((1, 1, 2, 2)), seed=11)
    comp.load_noise("b", torch.full((1, 1, 2, 2), 2.0), seed=22)

    comp.clear_noise("b")

    assert comp.seed_a == 11
    assert comp.seed_b is None
    assert not comp.has_both()
    assert torch.equal(comp.step(0.0), torch.ones((1, 1, 2, 2)))

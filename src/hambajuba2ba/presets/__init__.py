"""Preset loading utilities.

All presets are loaded from YAML files and cached for performance.
YAML allows comments, making the preset files self-documenting.
"""
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

PRESETS_DIR = Path(__file__).parent


def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML file from the presets directory."""
    with open(PRESETS_DIR / filename) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_envelope_presets() -> Dict[str, Dict[str, float]]:
    """Load envelope attack/release presets for each stem."""
    return _load_yaml("envelope.yaml")


@lru_cache(maxsize=1)
def load_brightness_presets() -> Dict[str, Dict[str, float]]:
    """Load brightness processing presets for each stem."""
    return _load_yaml("brightness.yaml")


@lru_cache(maxsize=1)
def load_dual_layer_presets() -> Dict[str, Dict[str, float]]:
    """Load flash/sustain dual-layer response presets."""
    return _load_yaml("dual_layer.yaml")


@lru_cache(maxsize=1)
def load_physics_presets() -> Dict[str, Dict[str, float]]:
    """Load mass-spring-damper physics presets."""
    return _load_yaml("physics.yaml")

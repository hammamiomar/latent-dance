"""Configuration module - single source of truth.

Usage:
    from hambajuba2ba.config import PipelineConfig

    # Default config with auto-detection
    config = PipelineConfig()

    # With env var overrides
    from hambajuba2ba.config import load_from_env
    config = load_from_env()

    # From YAML file + env overrides
    from hambajuba2ba.config import load_config
    config = load_config(Path("config.yaml"))
"""
from .base import PipelineConfig
from .server import ServerConfig
from .audio import AudioConfig
from .streaming import StreamingConfig
from .strategy import StrategyConfig
from .sae import SAEConfig
from .loader import load_config, load_from_env

__all__ = [
    "PipelineConfig",
    "ServerConfig",
    "AudioConfig",
    "StreamingConfig",
    "StrategyConfig",
    "SAEConfig",
    "load_config",
    "load_from_env",
]

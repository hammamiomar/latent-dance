"""Config loading utilities."""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base import PipelineConfig


def _set_nested(config: PipelineConfig, path: Tuple, value: str) -> None:
    """Set a (possibly nested) config value from an env-map path.

    Path grammar: attribute names, then an optional trailing converter —
    ("device",), ("height", int), ("server", "host"), ("server", "port", int).
    Strings are never callable, so a callable tail is always the converter.

    Args:
        config: PipelineConfig instance
        path: Attribute path per the grammar above
        value: String value from environment
    """
    if callable(path[-1]):
        value = path[-1](value)
        path = path[:-1]
    target = config
    for name in path[:-1]:
        target = getattr(target, name)
    setattr(target, path[-1], value)


def load_from_env(config: Optional[PipelineConfig] = None) -> PipelineConfig:
    """Override config values from environment variables.

    Env vars use HAMBAJUBA_ prefix:
    - HAMBAJUBA_SERVER_HOST=0.0.0.0
    - HAMBAJUBA_SERVER_PORT=9000
    - HAMBAJUBA_AUDIO_SAMPLE_RATE=48000
    - HAMBAJUBA_AUDIO_MAX_UPLOAD_MB=1000
    - HAMBAJUBA_STREAMING_FPS=60
    - HAMBAJUBA_STREAMING_JPEG_QUALITY=85
    - HAMBAJUBA_DEVICE=cuda
    - HAMBAJUBA_DTYPE=float16

    Args:
        config: Optional existing config to modify. Creates new if None.

    Returns:
        PipelineConfig with env overrides applied
    """
    if config is None:
        config = PipelineConfig()

    env_map: Dict[str, Tuple] = {
        # Server
        "HAMBAJUBA_SERVER_HOST": ("server", "host"),
        "HAMBAJUBA_SERVER_PORT": ("server", "port", int),
        "HAMBAJUBA_SERVER_RELOAD": ("server", "reload", lambda x: x.lower() == "true"),
        # Audio
        "HAMBAJUBA_AUDIO_SAMPLE_RATE": ("audio", "sample_rate", int),
        "HAMBAJUBA_AUDIO_MAX_UPLOAD_MB": ("audio", "max_upload_mb", int),
        "HAMBAJUBA_AUDIO_CACHE_TTL": ("audio", "cache_ttl_seconds", int),
        "HAMBAJUBA_AUDIO_DEFAULT_BPM": ("audio", "default_bpm", float),
        "HAMBAJUBA_AUDIO_FEATURE_LEVEL": ("audio", "feature_level"),
        "HAMBAJUBA_AUDIO_FEATURE_BACKEND": ("audio", "feature_backend"),
        "HAMBAJUBA_AUDIO_FEATURE_DEVICE": ("audio", "feature_device"),
        "HAMBAJUBA_AUDIO_COUPLING_STEMS": ("audio", "coupling_stems"),
        "HAMBAJUBA_AUDIO_FEATURE_CACHE": ("audio", "enable_feature_cache", lambda x: x.lower() == "true"),
        "HAMBAJUBA_AUDIO_SONG_LIBRARY_DIR": ("audio", "song_library_dir"),
        # Streaming
        "HAMBAJUBA_STREAMING_FPS": ("streaming", "fps", float),
        "HAMBAJUBA_STREAMING_JPEG_QUALITY": ("streaming", "jpeg_quality", int),
        "HAMBAJUBA_STREAMING_QUEUE_TIMEOUT": ("streaming", "queue_timeout", float),
        "HAMBAJUBA_STREAMING_LATE_TOLERANCE": ("streaming", "late_tolerance", float),
        # Top-level
        "HAMBAJUBA_DEVICE": ("device",),
        "HAMBAJUBA_DTYPE": ("dtype",),
        "HAMBAJUBA_HEIGHT": ("height", int),
        "HAMBAJUBA_WIDTH": ("width", int),
        "HAMBAJUBA_SEED": ("seed", int),
        "HAMBAJUBA_CPU_WORKERS": ("cpu_workers", int),
        "HAMBAJUBA_WARMUP_ITERATIONS": ("warmup_iterations", int),
        # SAE artifacts
        "HAMBAJUBA_SAE_WEIGHTS_DIR": ("sae", "weights_dir"),
        "HAMBA_SAE_WEIGHTS_DIR": ("sae", "weights_dir"),
        "HAMBAJUBA_SAE_AUTO_DOWNLOAD": ("sae", "auto_download_weights", lambda x: x.lower() == "true"),
        "HAMBA_ARTIFACT_REPO": ("sae", "artifact_repo_id"),
        "HAMBA_ARTIFACT_REPO_TYPE": ("sae", "artifact_repo_type"),
        "HAMBA_ARTIFACT_WEIGHTS_SUBDIR": ("sae", "artifact_weights_subdir"),
        "HAMBA_ARTIFACT_DIR": ("sae", "artifact_cache_dir"),
    }

    for env_key, path in env_map.items():
        if value := os.getenv(env_key):
            try:
                _set_nested(config, path, value)
            except (ValueError, AttributeError) as e:
                # Log warning but don't fail
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to apply env var {env_key}={value}: {e}"
                )

    # An overridden device invalidates the dtype derived at construction
    # (HAMBAJUBA_DEVICE=cpu on a CUDA box must not keep float16). An
    # explicit HAMBAJUBA_DTYPE still wins; resolve() is idempotent.
    if os.getenv("HAMBAJUBA_DEVICE") and not os.getenv("HAMBAJUBA_DTYPE"):
        config.dtype = None
    config.resolve()

    return config


def load_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """Load config, optionally from YAML file, then apply env overrides.

    Priority (highest to lowest):
    1. Environment variables (HAMBAJUBA_*)
    2. YAML config file (if provided)
    3. Default values

    Args:
        config_path: Optional path to YAML config file

    Returns:
        Fully resolved PipelineConfig
    """
    config = PipelineConfig()

    # Load from YAML if provided
    if config_path and config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                data = yaml.safe_load(f)
                if data:
                    config = _merge_dict_to_config(config, data)
        except ImportError:
            import logging

            logging.getLogger(__name__).warning(
                "PyYAML not installed, skipping config file loading"
            )
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Failed to load config file: {e}")

    # Apply env overrides (highest priority)
    return load_from_env(config)


def _merge_dict_to_config(config: PipelineConfig, data: Dict[str, Any]) -> PipelineConfig:
    """Merge a dictionary into a PipelineConfig.

    Handles nested configs like server.port, audio.sample_rate, etc.
    """
    for key, value in data.items():
        if hasattr(config, key):
            attr = getattr(config, key)
            if isinstance(value, dict) and hasattr(attr, "__dataclass_fields__"):
                # Nested config - recursively set attributes
                for nested_key, nested_value in value.items():
                    if hasattr(attr, nested_key):
                        setattr(attr, nested_key, nested_value)
            else:
                # Top-level attribute
                setattr(config, key, value)

    # Same invariant as the env path: a YAML device override without an
    # explicit dtype re-derives it (load_config resolves via load_from_env).
    if "device" in data and "dtype" not in data:
        config.dtype = None
    return config

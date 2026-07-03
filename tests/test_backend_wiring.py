"""Tests for backend wiring and strategy factory.

Verifies:
1. Strategy factory creates correct strategy for each mode
2. Type hints and dependencies return correct types
3. App configuration is correct
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from app.caching import CacheManager
from app.strategies import SAESteeringStrategy, create_strategy


class TestStrategyFactory:
    """Tests for the strategy factory function."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline."""
        pipeline = MagicMock()
        pipeline.encode_prompt.return_value = (MagicMock(), MagicMock())
        pipeline._base_latent = MagicMock()
        return pipeline

    @pytest.fixture
    def mock_websocket(self):
        return MagicMock()

    @pytest.fixture
    def mock_config(self):
        """No spec= constraint — strategy reads many nested config attrs."""
        return MagicMock()

    @pytest.fixture
    def mock_audio_cache(self):
        return MagicMock(spec=CacheManager)

    @pytest.fixture
    def mock_executors(self):
        """Provide GPU lock and CPU executor for strategy construction."""
        return {
            "gpu_lock": asyncio.Lock(),
            "cpu_executor": ThreadPoolExecutor(max_workers=1),
        }

    def test_sae_steering_mode_returns_strategy(
        self, mock_pipeline, mock_config, mock_websocket, mock_audio_cache, mock_executors
    ):
        """Test that 'sae_steering' mode returns SAESteeringStrategy."""
        strategy = create_strategy(
            mode="sae_steering",
            pipeline=mock_pipeline,
            config=mock_config,
            websocket=mock_websocket,
            audio_cache=mock_audio_cache,
            **mock_executors,
        )
        assert isinstance(strategy, SAESteeringStrategy)

    def test_unknown_mode_raises_error(
        self, mock_pipeline, mock_config, mock_websocket, mock_audio_cache, mock_executors
    ):
        """Test that unknown modes raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            create_strategy(
                mode="looping",
                pipeline=mock_pipeline,
                config=mock_config,
                websocket=mock_websocket,
                audio_cache=mock_audio_cache,
                **mock_executors,
            )

    def test_old_fourcorner_mode_raises_error(
        self, mock_pipeline, mock_config, mock_websocket, mock_audio_cache, mock_executors
    ):
        with pytest.raises(ValueError, match="Unknown mode"):
            create_strategy(
                mode="fourcorner",
                pipeline=mock_pipeline,
                config=mock_config,
                websocket=mock_websocket,
                audio_cache=mock_audio_cache,
                **mock_executors,
            )

    def test_error_message_mentions_sae_steering(
        self, mock_pipeline, mock_config, mock_websocket, mock_audio_cache, mock_executors
    ):
        with pytest.raises(ValueError, match="sae_steering"):
            create_strategy(
                mode="invalid",
                pipeline=mock_pipeline,
                config=mock_config,
                websocket=mock_websocket,
                audio_cache=mock_audio_cache,
                **mock_executors,
            )


class TestDependencyTypes:
    """Tests for dependency injection return types."""

    def test_dependencies_have_no_backend_coupling(self):
        """Phase 2: dependencies serve the active backend generically."""
        from app import dependencies
        assert not hasattr(dependencies, "SAESteerablePipeline")
        assert hasattr(dependencies, "get_capabilities_ws")
        assert hasattr(dependencies, "get_server_mode_ws")

    def test_strategies_module_exports_sae_strategy(self):
        from app import strategies
        assert hasattr(strategies, "SAESteeringStrategy")
        assert strategies.SAESteeringStrategy is SAESteeringStrategy


class TestMainAppConfiguration:
    """Tests for main.py app configuration.

    NOTE: These tests do source-level string matching. They are fragile
    and exist as migration guards, not behavioral tests. Consider
    replacing with runtime tests when possible.
    """

    def test_main_has_no_hardcoded_backend(self):
        """Phase 2: main resolves the pipeline through the backend registry."""
        with open("app/main.py") as f:
            content = f.read()
        assert "SAESteerablePipeline" not in content
        assert "HAMBA_MODE" in content
        assert "get_backend" in content

    def test_main_has_lifespan_handler(self):
        with open("app/main.py") as f:
            content = f.read()
        assert "async def lifespan" in content
        assert "@asynccontextmanager" in content

    def test_main_includes_streaming_router(self):
        with open("app/main.py") as f:
            content = f.read()
        assert "streaming.router" in content

    def test_main_includes_audio_router(self):
        with open("app/main.py") as f:
            content = f.read()
        assert "audio.router" in content

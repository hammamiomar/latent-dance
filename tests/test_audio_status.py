"""Tests for bounded audio processing status bookkeeping."""

import asyncio

from app.routers import audio


def test_processing_status_expires_and_stays_bounded():
    async def scenario() -> None:
        original_ttl = audio._PROCESSING_STATUS_TTL_SECONDS
        original_max = audio._PROCESSING_STATUS_MAX_ENTRIES
        try:
            audio._PROCESSING_STATUS_TTL_SECONDS = 60
            audio._PROCESSING_STATUS_MAX_ENTRIES = 2
            async with audio._status_lock:
                audio._processing_status.clear()

            await audio._set_status("a", {"status": "processing"})
            await audio._set_status("b", {"status": "processing"})
            await audio._set_status("c", {"status": "processing"})

            async with audio._status_lock:
                assert len(audio._processing_status) == 2
                assert "a" not in audio._processing_status

            audio._PROCESSING_STATUS_TTL_SECONDS = 0
            await audio._set_status("expired", {"status": "error"})
            assert await audio._get_status("expired") is None
        finally:
            audio._PROCESSING_STATUS_TTL_SECONDS = original_ttl
            audio._PROCESSING_STATUS_MAX_ENTRIES = original_max
            async with audio._status_lock:
                audio._processing_status.clear()

    asyncio.run(scenario())

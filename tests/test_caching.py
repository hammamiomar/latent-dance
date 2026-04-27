"""Tests for the CacheManager class."""

import time

from app.caching import CacheManager


class TestCacheManager:
    """Test suite for CacheManager."""

    def test_init(self):
        """Test cache manager initialization."""
        cache = CacheManager(default_ttl=60)
        assert cache.default_ttl == 60
        assert len(cache) == 0

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = CacheManager()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self):
        """Test getting a key that doesn't exist."""
        cache = CacheManager()
        assert cache.get("nonexistent") is None

    def test_custom_ttl(self):
        """Test setting a value with custom TTL."""
        cache = CacheManager(default_ttl=60)
        cache.set("key1", "value1", ttl=1)  # 1 second TTL

        # Value should exist immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Value should be expired and purged
        assert cache.get("key1") is None

    def test_default_ttl(self):
        """Test that default TTL is used when not specified."""
        cache = CacheManager(default_ttl=1)
        cache.set("key1", "value1")

        # Value should exist immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Value should be expired
        assert cache.get("key1") is None

    def test_delete(self):
        """Test manual deletion."""
        cache = CacheManager()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Delete and verify
        assert cache.delete("key1") is True
        assert cache.get("key1") is None

        # Delete non-existent key
        assert cache.delete("key1") is False

    def test_clear(self):
        """Test clearing all cache entries."""
        cache = CacheManager()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert len(cache) == 3

        cache.clear()
        assert len(cache) == 0
        assert cache.get("key1") is None

    def test_cleanup_expired(self):
        """Test manual cleanup of expired entries."""
        cache = CacheManager(default_ttl=1)
        cache.set("key1", "value1")
        cache.set("key2", "value2", ttl=10)  # Won't expire
        cache.set("key3", "value3")

        # Wait for some to expire
        time.sleep(1.1)

        # Manually trigger cleanup
        removed = cache.cleanup_expired()

        # Should have removed 2 expired entries
        assert removed == 2
        assert len(cache) == 1
        assert cache.get("key2") == "value2"

    def test_contains(self):
        """Test __contains__ operator."""
        cache = CacheManager(default_ttl=1)
        cache.set("key1", "value1")

        assert "key1" in cache
        assert "key2" not in cache

        # Wait for expiration
        time.sleep(1.1)

        # Expired keys should not be "in" cache
        assert "key1" not in cache

    def test_complex_values(self):
        """Test storing complex data structures."""
        cache = CacheManager()

        # Dictionary
        data = {"features": [1, 2, 3], "timestamps": [0.0, 0.1, 0.2]}
        cache.set("audio1", data)
        assert cache.get("audio1") == data

        # List
        cache.set("list", [1, 2, 3, 4, 5])
        assert cache.get("list") == [1, 2, 3, 4, 5]

        # Nested structure
        nested = {"a": {"b": {"c": "value"}}}
        cache.set("nested", nested)
        assert cache.get("nested") == nested

    def test_overwrite(self):
        """Test overwriting an existing key."""
        cache = CacheManager()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"

    def test_ttl_extension(self):
        """Test that re-setting a key extends its TTL."""
        cache = CacheManager()
        cache.set("key1", "value1", ttl=1)

        # Wait half the TTL
        time.sleep(0.5)

        # Re-set with new TTL
        cache.set("key1", "value1", ttl=2)

        # Wait original TTL
        time.sleep(0.6)

        # Should still exist (new TTL is 2 seconds from re-set)
        assert cache.get("key1") == "value1"

    def test_concurrent_expiration(self):
        """Test that multiple entries can expire correctly."""
        cache = CacheManager()
        cache.set("key1", "value1", ttl=1)
        cache.set("key2", "value2", ttl=1)
        cache.set("key3", "value3", ttl=2)

        time.sleep(1.1)

        # First two should be expired
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

        time.sleep(1.0)

        # All should be expired
        assert cache.get("key3") is None

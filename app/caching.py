"""
Managed caching with TTL (Time-To-Live) eviction policy.

This module provides a thread-safe cache manager for storing temporary data
with automatic expiration. Used primarily for audio feature caching.
"""

import threading
import time
from typing import Any, Dict, Tuple


class CacheManager:
    """
    A time-aware cache with automatic TTL-based eviction.

    Stores items with expiration timestamps and automatically purges
    expired entries on access. Thread-safe for both sync and async contexts
    via internal locking.
    """

    def __init__(self, default_ttl: int = 3600):
        """
        Initialize the cache manager.

        Args:
            default_ttl: Default time-to-live in seconds (default: 3600 = 1 hour)
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set a value in the cache with a specific TTL.

        Args:
            key: Cache key
            value: Value to store (any type)
            ttl: Time-to-live in seconds (uses default_ttl if None)
        """
        if ttl is None:
            ttl = self.default_ttl

        expires_at = time.time() + ttl
        with self._lock:
            self._cache[key] = (value, expires_at)

    def get(self, key: str) -> Any | None:
        """
        Get a value from the cache.

        Returns None if the key doesn't exist or has expired.
        Automatically purges expired items on access.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                return None

            value, expires_at = self._cache[key]

            # Check expiration
            if time.time() > expires_at:
                # Purge expired item
                del self._cache[key]
                return None

            return value

    def delete(self, key: str) -> bool:
        """
        Manually delete a key from the cache.

        Args:
            key: Cache key

        Returns:
            True if key was present and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """
        Manually trigger cleanup of all expired items.

        Returns:
            Number of items removed
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                key for key, (_, expires_at) in self._cache.items()
                if now > expires_at
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    def __len__(self) -> int:
        """Return the number of items currently in the cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if a non-expired key exists in the cache."""
        return self.get(key) is not None

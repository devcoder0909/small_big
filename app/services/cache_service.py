"""In-memory TTL cache service for API responses."""

import time
from functools import wraps
from typing import Any
from app.core import get_settings


class CacheService:
    """Simple in-memory TTL cache for API endpoint responses."""

    def __init__(self):
        self._cache: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        """Get a cached value if not expired."""
        if key in self._cache:
            value, expires_at = self._cache[key]
            if time.monotonic() < expires_at:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int):
        """Set a cache value with TTL in seconds."""
        self._cache[key] = (value, time.monotonic() + ttl)

    def invalidate(self, key: str):
        """Remove a specific cache entry."""
        self._cache.pop(key, None)

    def clear(self):
        """Clear the entire cache."""
        self._cache.clear()

    def cleanup(self):
        """Remove expired entries."""
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._cache.items() if now >= exp]
        for k in expired:
            del self._cache[k]


# Global cache instance
cache = CacheService()

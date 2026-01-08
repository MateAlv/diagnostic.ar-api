from __future__ import annotations

import json
import logging
from typing import Any, Optional

import redis

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, url: str, ttl_seconds: int, enabled: bool = True) -> None:
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self._client = None
        if enabled:
            try:
                self._client = redis.Redis.from_url(url, decode_responses=True)
                self._client.ping()
            except redis.RedisError as exc:
                logger.warning("redis unavailable, caching disabled: %s", exc)
                self.enabled = False
                self._client = None

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled or not self._client:
            return None
        try:
            payload = self._client.get(key)
        except redis.RedisError as exc:
            logger.warning("redis get failed: %s", exc)
            return None
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled or not self._client:
            return
        try:
            self._client.setex(key, self.ttl_seconds, json.dumps(value))
        except redis.RedisError as exc:
            logger.warning("redis set failed: %s", exc)

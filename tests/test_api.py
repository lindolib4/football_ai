from __future__ import annotations

from api.cache import JsonFileCache


def test_cache_key_stable() -> None:
    cache = JsonFileCache(ttl_sec=3600)
    key1 = cache.make_key("/league-list", {"a": 1, "b": 2})
    key2 = cache.make_key("/league-list", {"b": 2, "a": 1})
    assert key1 == key2

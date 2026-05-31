"""Model pool — multi-model lifecycle management.

EnginePool with LRU eviction, pinning, TTL auto-unload, and
4-tier ProcessMemoryEnforcer (safe/balanced/aggressive/custom).
"""

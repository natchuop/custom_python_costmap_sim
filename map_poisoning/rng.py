"""Stable independent random streams derived from the master seed."""
from __future__ import annotations
import hashlib
import random

def derived_seed(master_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"map-poisoning/v1/{master_seed}/{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")

def named_rng(master_seed: int, name: str) -> random.Random:
    return random.Random(derived_seed(master_seed, name))

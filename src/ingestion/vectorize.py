from __future__ import annotations

import hashlib
import math
import re


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def hashing_vector(text: str, dimension: int) -> list[float]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")

    vector = [0.0 for _ in range(dimension)]
    tokens = tokenize_words(text)
    if not tokens:
        return vector

    for token in tokens:
        index = _stable_index(token=token, dimension=dimension)
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _stable_index(*, token: str, dimension: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % dimension

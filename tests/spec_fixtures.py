"""Callables referenced by declarative pipeline specs in tests."""

from typing import Iterable


def inc(x: int) -> int:
    return x + 1


def double(x: int) -> int:
    return x * 2


def sum_all(values: Iterable[int]) -> int:
    return sum(values)

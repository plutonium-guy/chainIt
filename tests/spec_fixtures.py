"""Callables referenced by declarative pipeline specs in tests."""


def inc(x):
    return x + 1


def double(x):
    return x * 2


def sum_all(values):
    return sum(values)

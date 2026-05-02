"""Shared utilities for translation providers."""
from typing import Iterator, TypeVar

T = TypeVar("T")


def batches(items: list[T], n: int) -> Iterator[list[T]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]

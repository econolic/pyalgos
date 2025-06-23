"""Heap data structures package.

This package provides implementations of heap data structures with comprehensive
type safety, performance optimization, and visualization capabilities.
"""

from __future__ import annotations

try:
    from pyalgos.heaps.binary_heap import BinaryHeap
except ImportError:
    from .binary_heap import BinaryHeap

__all__ = ["BinaryHeap"]

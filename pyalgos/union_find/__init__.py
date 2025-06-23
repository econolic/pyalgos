"""
Union-Find (Disjoint Set Union) data structures.

This module provides efficient union-find data structures with path compression
and union by rank/size optimizations for dynamic connectivity problems.
"""

from .disjoint_set_union import DisjointSetUnion

__all__ = ["DisjointSetUnion"]

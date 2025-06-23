"""Tree data structures package.

This package provides implementations of tree data structures including
segment trees for range queries and tries for string operations.
"""

from __future__ import annotations

try:
    from pyalgos.trees.segment_tree import SegmentTree
    from pyalgos.trees.trie import Trie
except ImportError:
    from .segment_tree import SegmentTree
    from .trie import Trie

__all__ = ["SegmentTree", "Trie"]

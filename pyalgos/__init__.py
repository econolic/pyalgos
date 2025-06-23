"""
PyAlgos: A high-quality Python library of fundamental data structures and algorithms.

This library provides modern, well-documented, and rigorously tested implementations
of essential data structures and algorithms. All implementations follow PEP 8 style
guidelines, include comprehensive type hints, and are designed for both educational
and practical use.

Modules:
    linear: Linear data structures (Stack, Queue, LinkedList)
    heaps: Heap data structures (BinaryHeap)
    trees: Tree data structures (SegmentTree)
    strings: String data structures (Trie)
    union_find: Union-Find data structures (DisjointSetUnion)
    graphs: Graph data structures and algorithms (Graph, BFS, DFS)
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "PyAlgos Contributors"
__email__ = "pyalgos@example.com"
__license__ = "MIT"

# Re-export main classes for convenient imports
__all__ = []

# Import linear structures
try:
    from pyalgos.linear.stack import Stack
    from pyalgos.linear.queue import Queue
    from pyalgos.linear.linked_list import LinkedList
    __all__.extend(["Stack", "Queue", "LinkedList"])
except ImportError:
    pass

# Import heaps
try:
    from pyalgos.heaps.binary_heap import BinaryHeap
    __all__.append("BinaryHeap")
except ImportError:
    pass

# Import trees
try:
    from pyalgos.trees.segment_tree import SegmentTree
    __all__.append("SegmentTree")
except ImportError:
    pass

# Import string structures  
try:
    from pyalgos.strings.trie import Trie
    __all__.append("Trie")
except ImportError:
    pass

# Import union-find structures
try:
    from pyalgos.union_find.disjoint_set_union import DisjointSetUnion, GenericDisjointSetUnion
    __all__.extend(["DisjointSetUnion", "GenericDisjointSetUnion"])
except ImportError:
    pass

# Import graph structures
try:
    from pyalgos.graphs.graph import Graph
    from pyalgos.graphs.traversal import BFS, DFS
    __all__.extend(["Graph", "BFS", "DFS"])
except ImportError:
    pass

# Export exceptions and types
try:
    from pyalgos.exceptions import (
        PyAlgosError,
        EmptyStructureError, 
        IndexOutOfBoundsError,
        InvalidOperationError
    )
    __all__.extend([
        "PyAlgosError",
        "EmptyStructureError", 
        "IndexOutOfBoundsError",
        "InvalidOperationError"
    ])
except ImportError:
    pass

"""
Test suite for SegmentTree data structure.
"""

import pytest
from pyalgos.trees.segment_tree import SegmentTree


class TestSegmentTreeBasicOperations:
    """Test basic segment tree operations."""

    def test_sum_tree_creation(self) -> None:
        """Test creating a sum segment tree."""
        data = [1, 3, 5, 7, 9]
        tree = SegmentTree.create_sum_tree(data)
        
        assert tree.size() == len(data)

    def test_range_queries(self) -> None:
        """Test range query operations."""
        data = [1, 3, 5, 7, 9]
        tree = SegmentTree.create_sum_tree(data)
        
        # Test sum queries
        assert tree.range_query(0, 4) == sum(data)
        assert tree.range_query(1, 3) == 3 + 5 + 7

    def test_point_updates(self) -> None:
        """Test point update operations."""
        data = [1, 3, 5, 7, 9]
        tree = SegmentTree.create_sum_tree(data)
        
        # Update value at index 2
        tree.update(2, 10)
        
        # Check that queries reflect the update
        assert tree.range_query(2, 2) == 10
        assert tree.range_query(0, 4) == 1 + 3 + 10 + 7 + 9

    def test_min_tree(self) -> None:
        """Test min segment tree."""
        data = [5, 2, 8, 1, 9]
        tree = SegmentTree.create_min_tree(data)
        
        assert tree.range_query(0, 4) == 1
        assert tree.range_query(0, 2) == 2

    def test_max_tree(self) -> None:
        """Test max segment tree."""
        data = [5, 2, 8, 1, 9]
        tree = SegmentTree.create_max_tree(data)
        
        assert tree.range_query(0, 4) == 9
        assert tree.range_query(1, 3) == 8

    def test_string_representation(self) -> None:
        """Test string representation."""
        data = [1, 2, 3]
        tree = SegmentTree.create_sum_tree(data)
        
        repr_str = repr(tree)
        assert "SegmentTree" in repr_str

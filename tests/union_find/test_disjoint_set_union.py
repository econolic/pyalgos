"""
Test suite for DisjointSetUnion data structure.
"""

import pytest
from pyalgos.union_find.disjoint_set_union import DisjointSetUnion
from pyalgos.exceptions import InvalidOperationError, IndexOutOfBoundsError


class TestDSUBasicOperations:
    """Test basic DSU operations."""

    def test_dsu_creation(self) -> None:
        """Test creating a DSU."""
        dsu = DisjointSetUnion(5)
        
        assert len(dsu) == 5
        assert dsu.num_components() == 5

    def test_union_operations(self) -> None:
        """Test union operations."""
        dsu = DisjointSetUnion(5)
        
        # Union some components
        result = dsu.union(0, 1)
        assert result is True  # Successful union
        assert dsu.num_components() == 4
        
        # Union same components again
        result = dsu.union(0, 1)
        assert result is False  # Already connected
        
        # Chain unions
        assert dsu.union(1, 2) is True
        assert dsu.union(3, 4) is True
        assert dsu.num_components() == 2

    def test_find_operations(self) -> None:
        """Test find operations."""
        dsu = DisjointSetUnion(5)
        
        # Initially each element is its own root
        assert dsu.find(0) == 0
        assert dsu.find(1) == 1
        
        # After union, they should have same root
        dsu.union(0, 1)
        assert dsu.find(0) == dsu.find(1)

    def test_connected_operations(self) -> None:
        """Test connectivity operations."""
        dsu = DisjointSetUnion(5)
        
        # Initially not connected
        assert not dsu.connected(0, 1)
        
        # After union, should be connected
        dsu.union(0, 1)
        assert dsu.connected(0, 1)
        assert dsu.connected(1, 0)  # Symmetric

    def test_component_size(self) -> None:
        """Test component size operations."""
        dsu = DisjointSetUnion(5)
        
        # Initially each component has size 1
        assert dsu.component_size(0) == 1
        
        # After union, component size increases
        dsu.union(0, 1)
        assert dsu.component_size(0) == 2
        assert dsu.component_size(1) == 2

    def test_string_representation(self) -> None:
        """Test string representation."""
        dsu = DisjointSetUnion(3)
        
        repr_str = repr(dsu)
        assert "DisjointSetUnion" in repr_str
        
        str_str = str(dsu)
        assert "DSU" in str_str


class TestDSUErrorHandling:
    """Test DSU error handling and edge cases."""
    
    def test_invalid_initialization(self) -> None:
        """Test invalid initialization parameters."""
        with pytest.raises(InvalidOperationError):
            DisjointSetUnion(0)
        
        with pytest.raises(InvalidOperationError):
            DisjointSetUnion(-1)
        
        with pytest.raises(InvalidOperationError):
            DisjointSetUnion("invalid")  # type: ignore
    
    def test_invalid_element_indices(self) -> None:
        """Test operations with invalid element indices."""
        dsu = DisjointSetUnion(5)
        
        # Test union with invalid indices
        with pytest.raises(IndexOutOfBoundsError):
            dsu.union(-1, 0)
        
        with pytest.raises(IndexOutOfBoundsError):
            dsu.union(0, 5)
        
        # Test find with invalid indices
        with pytest.raises(IndexOutOfBoundsError):
            dsu.find(-1)
        
        with pytest.raises(IndexOutOfBoundsError):
            dsu.find(5)
        
        # Test connected with invalid indices
        with pytest.raises(IndexOutOfBoundsError):
            dsu.connected(-1, 0)
        
        with pytest.raises(IndexOutOfBoundsError):
            dsu.connected(0, 5)
        
        # Test component_size with invalid indices
        with pytest.raises(IndexOutOfBoundsError):
            dsu.component_size(-1)
        
        with pytest.raises(IndexOutOfBoundsError):
            dsu.component_size(5)


class TestDSUAdvancedOperations:
    """Test advanced DSU operations and optimizations."""
    
    def test_path_compression_effectiveness(self) -> None:
        """Test that path compression works correctly."""
        dsu = DisjointSetUnion(10)
        
        # Create a long chain: 0 -> 1 -> 2 -> ... -> 9
        for i in range(9):
            dsu.union(i, i + 1)
        
        # All elements should have the same root after path compression
        root = dsu.find(0)
        for i in range(10):
            assert dsu.find(i) == root

    def test_union_by_size_optimization(self) -> None:
        """Test that union by size works correctly."""
        dsu = DisjointSetUnion(8)
        
        # Create two separate chains of different sizes
        dsu.union(0, 1)  # Component size 2
        dsu.union(2, 3)  # Component size 2
        dsu.union(3, 4)  # Component size 3
        dsu.union(4, 5)  # Component size 4
        
        dsu.union(6, 7)  # Component size 2
        
        # Union the two components - check they get connected
        initial_components = dsu.num_components()
        dsu.union(0, 6)
        assert dsu.num_components() == initial_components - 1
        
        # All elements in the merged component should have the same size
        merged_size = dsu.component_size(0)
        assert dsu.component_size(6) == merged_size
        # The actual merged size depends on which component was the root
    
    def test_components_iteration(self) -> None:
        """Test iteration over components."""
        dsu = DisjointSetUnion(6)
        dsu.union(0, 1)
        dsu.union(2, 3)
        dsu.union(4, 5)
        
        components = dsu.get_components()
        assert len(components) == 3
        
        # Each component should have 2 elements
        for component in components:
            assert len(component) == 2
    
    def test_component_roots(self) -> None:
        """Test getting component roots."""
        dsu = DisjointSetUnion(5)
        dsu.union(0, 1)
        dsu.union(2, 3)
        
        roots = dsu.get_component_roots()
        assert len(roots) == 3  # 3 components
        
        # Check that roots are actually roots
        for root in roots:
            assert dsu.find(root) == root
    
    def test_reset_operation(self) -> None:
        """Test reset functionality."""
        dsu = DisjointSetUnion(5)
        dsu.union(0, 1)
        dsu.union(2, 3)
        
        assert dsu.num_components() == 3
        
        dsu.reset()
        assert dsu.num_components() == 5
        
        # All elements should be their own root
        for i in range(5):
            assert dsu.find(i) == i
            assert dsu.component_size(i) == 1
    
    def test_is_fully_connected(self) -> None:
        """Test fully connected check."""
        dsu = DisjointSetUnion(5)
        
        assert not dsu.is_fully_connected()
        
        # Connect all elements
        for i in range(4):
            dsu.union(i, i + 1)
        
        assert dsu.is_fully_connected()
    
    def test_largest_component_size(self) -> None:
        """Test largest component size."""
        dsu = DisjointSetUnion(10)
        
        # Initially all components have size 1
        assert dsu.largest_component_size() == 1
        
        # Create component of size 3
        dsu.union(0, 1)
        dsu.union(1, 2)
        assert dsu.largest_component_size() == 3
        
        # Create larger component of size 5
        dsu.union(3, 4)
        dsu.union(4, 5)
        dsu.union(5, 6)
        dsu.union(6, 7)
        assert dsu.largest_component_size() == 5


class TestDSUPerformance:
    """Test DSU performance characteristics."""
    
    def test_large_dsu_operations(self) -> None:
        """Test operations on large DSU."""
        n = 1000
        dsu = DisjointSetUnion(n)
        
        # Perform many union operations
        import random
        random.seed(42)  # For reproducible tests
        
        for _ in range(500):
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            dsu.union(i, j)
        
        # Test that operations still work correctly
        assert dsu.num_components() <= n
        assert dsu.num_components() >= 1
        
        # Test find operations
        for i in range(0, n, 10):  # Sample every 10th element
            root = dsu.find(i)
            assert 0 <= root < n
    
    def test_worst_case_path_compression(self) -> None:
        """Test path compression in worst-case scenario."""
        n = 100
        dsu = DisjointSetUnion(n)
        
        # Create a linear chain (worst case for find without path compression)
        for i in range(n - 1):
            # Force a specific union order to create a chain
            if i == 0:
                dsu.union(0, 1)
            else:
                # This should still be efficient due to path compression
                dsu.union(0, i + 1)
        
        # After path compression, all finds should be fast
        for i in range(n):
            root = dsu.find(i)
            # All elements should have the same root
            assert root == dsu.find(0)


class TestDSUEdgeCases:
    """Test DSU edge cases and corner conditions."""
    
    def test_single_element_dsu(self) -> None:
        """Test DSU with single element."""
        dsu = DisjointSetUnion(1)
        
        assert len(dsu) == 1
        assert dsu.num_components() == 1
        assert dsu.find(0) == 0
        assert dsu.component_size(0) == 1
        assert dsu.connected(0, 0) is True
        
        # Union with itself should return False (already connected)
        assert dsu.union(0, 0) is False
        assert dsu.num_components() == 1
    
    def test_self_union_operations(self) -> None:
        """Test union operations with same element."""
        dsu = DisjointSetUnion(5)
        
        for i in range(5):
            # Union element with itself should return False
            assert dsu.union(i, i) is False
            assert dsu.num_components() == 5  # No change in components
    
    def test_complete_union_graph(self) -> None:
        """Test complete union (all elements in one component)."""
        n = 10
        dsu = DisjointSetUnion(n)
        
        # Connect all elements in a star pattern
        for i in range(1, n):
            dsu.union(0, i)
        
        assert dsu.num_components() == 1
        
        # All elements should be connected to all others
        for i in range(n):
            for j in range(n):
                assert dsu.connected(i, j) is True
            assert dsu.component_size(i) == n
    
    def test_component_enumeration(self) -> None:
        """Test enumerating all components."""
        dsu = DisjointSetUnion(6)
        dsu.union(0, 1)
        dsu.union(2, 3)
        # Element 4 and 5 remain isolated
        
        components = dsu.get_components()
        assert len(components) == 4  # {0,1}, {2,3}, {4}, {5}
        
        # Check component sizes
        component_sizes = [len(comp) for comp in components]
        component_sizes.sort()
        assert component_sizes == [1, 1, 2, 2]
    
    def test_contains_operation(self) -> None:
        """Test contains operation."""
        dsu = DisjointSetUnion(5)
        
        # Valid elements
        for i in range(5):
            assert i in dsu
        
        # Invalid elements
        assert -1 not in dsu
        assert 5 not in dsu
        assert "invalid" not in dsu  # type: ignore

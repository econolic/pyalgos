"""
Test suite for the LinkedList data structure.

This module provides testing of the LinkedList implementation including
edge cases, performance validation, and error handling.
"""

import pytest
from typing import List
import time

from pyalgos.linear.linked_list import LinkedList
from pyalgos.exceptions import EmptyStructureError, InvalidOperationError, IndexOutOfBoundsError


class TestLinkedListBasicOperations:
    """Test basic linked list operations and functionality."""

    def test_empty_list_creation(self) -> None:
        """Test creating an empty linked list."""
        ll = LinkedList[int]()
        assert len(ll) == 0
        assert ll.is_empty()

    def test_append_operations(self) -> None:
        """Test append operations."""
        ll = LinkedList[int]()
        
        ll.append(1)
        assert len(ll) == 1
        assert ll.get(0) == 1
        
        ll.append(2)
        ll.append(3)
        assert len(ll) == 3
        assert ll.get(0) == 1
        assert ll.get(1) == 2
        assert ll.get(2) == 3

    def test_prepend_operations(self) -> None:
        """Test prepend operations."""
        ll = LinkedList[int]()
        
        ll.prepend(1)
        assert len(ll) == 1
        assert ll.get(0) == 1
        
        ll.prepend(2)
        ll.prepend(3)
        assert len(ll) == 3
        assert ll.get(0) == 3  # Most recently prepended
        assert ll.get(1) == 2
        assert ll.get(2) == 1

    def test_insert_at_index(self) -> None:
        """Test insert at specific index."""
        ll = LinkedList[int]()
        
        # Insert into empty list
        ll.insert(0, 1)
        assert ll.get(0) == 1
        
        # Insert at beginning
        ll.insert(0, 0)
        assert ll.get(0) == 0
        assert ll.get(1) == 1
        
        # Insert at end
        ll.insert(2, 2)
        assert ll.get(2) == 2
        
        # Insert in middle
        ll.insert(1, 10)
        assert ll.get(1) == 10
        assert len(ll) == 4

    def test_get_operations(self) -> None:
        """Test get operations."""
        ll = LinkedList[str]()
        items = ["first", "second", "third"]
        
        for item in items:
            ll.append(item)
        
        # Valid indices
        for i, item in enumerate(items):
            assert ll.get(i) == item        # Invalid indices - actually out of bounds
        with pytest.raises(IndexOutOfBoundsError):
            ll.get(-4)  # Too negative for size 3
        
        with pytest.raises(IndexOutOfBoundsError):
            ll.get(len(items))  # 3 is out of bounds for indices 0,1,2
        
        # Empty list
        empty_ll = LinkedList[int]()
        with pytest.raises(IndexOutOfBoundsError):
            empty_ll.get(0)

    def test_set_operations(self) -> None:
        """Test set operations."""
        ll = LinkedList[int]()
        ll.append(1)
        ll.append(2)
        ll.append(3)
        
        # Valid set operations
        ll.set(0, 10)
        ll.set(1, 20)
        ll.set(2, 30)
        
        assert ll.get(0) == 10
        assert ll.get(1) == 20
        assert ll.get(2) == 30        # Invalid indices - actually out of bounds
        with pytest.raises(IndexOutOfBoundsError):
            ll.set(-4, 100)  # Too negative for size 3
        
        with pytest.raises(IndexOutOfBoundsError):
            ll.set(3, 100)  # Index 3 is out of bounds for indices 0,1,2

    def test_remove_operations(self) -> None:
        """Test remove operations."""
        ll = LinkedList[int]()
        for i in range(5):
            ll.append(i)
        
        # Remove from middle
        removed = ll.remove(2)
        assert removed == 2
        assert len(ll) == 4
        assert ll.get(2) == 3  # Next element shifted
        
        # Remove from beginning
        removed = ll.remove(0)
        assert removed == 0
        assert ll.get(0) == 1
        
        # Remove from end
        removed = ll.remove(len(ll) - 1)
        assert removed == 4
        assert len(ll) == 2
        
        # Invalid remove
        with pytest.raises(InvalidOperationError):
            ll.remove(10)

    def test_remove_by_value(self) -> None:
        """Test remove by value operations."""
        ll = LinkedList[str]()
        items = ["apple", "banana", "cherry", "banana"]
        
        for item in items:
            ll.append(item)
        
        # Remove existing value
        assert ll.remove_value("banana") == True
        assert len(ll) == 3
        assert ll.get(1) == "cherry"  # First "banana" was removed
        
        # Remove non-existing value
        assert ll.remove_value("grape") == False
        assert len(ll) == 3

    def test_find_operations(self) -> None:
        """Test find operations."""
        ll = LinkedList[int]()
        items = [10, 20, 30, 20, 40]
        
        for item in items:
            ll.append(item)
        
        # Find existing values
        assert ll.find(10) == 0
        assert ll.find(20) == 1  # First occurrence
        assert ll.find(40) == 4
        
        # Find non-existing value
        assert ll.find(50) == -1

    def test_reverse_operations(self) -> None:
        """Test reverse operations."""
        ll = LinkedList[int]()
        items = [1, 2, 3, 4, 5]
        
        for item in items:
            ll.append(item)
        
        ll.reverse()
        
        # Check reversed order
        reversed_items = [5, 4, 3, 2, 1]
        for i, item in enumerate(reversed_items):
            assert ll.get(i) == item
        
        # Test reverse on empty list
        empty_ll = LinkedList[int]()
        empty_ll.reverse()  # Should not raise error
        assert empty_ll.is_empty()
          # Test reverse on single element
        single_ll = LinkedList[int]()
        single_ll.append(42)
        single_ll.reverse()
        assert single_ll.get(0) == 42

    def test_clear_operations(self) -> None:
        """Test clear operations."""
        ll = LinkedList[int]()
        for i in range(10):
            ll.append(i)
        
        assert len(ll) == 10
        
        ll.clear()
        
        assert len(ll) == 0
        assert ll.is_empty()
        
        # Should be able to add after clear
        ll.append(42)
        assert ll.get(0) == 42


class TestLinkedListAdvancedOperations:
    """Test advanced linked list operations."""

    def test_to_list_conversion(self) -> None:
        """Test conversion to Python list."""
        ll = LinkedList[str]()
        items = ["a", "b", "c", "d"]
        
        for item in items:
            ll.append(item)
        
        result_list = ll.to_list()
        assert result_list == items
        
        # Test empty list
        empty_ll = LinkedList[int]()
        assert empty_ll.to_list() == []

    def test_from_list_creation(self) -> None:
        """Test creation from Python list."""
        items = [1, 2, 3, 4, 5]
        ll = LinkedList.from_list(items)
        
        assert len(ll) == 5
        for i, item in enumerate(items):
            assert ll.get(i) == item

    def test_iteration(self) -> None:
        """Test iteration over linked list."""
        ll = LinkedList[int]()
        items = [10, 20, 30, 40]
        
        for item in items:
            ll.append(item)
        
        # Test forward iteration
        result = []
        for item in ll:
            result.append(item)
        assert result == items
        
        # Test with list comprehension
        squared = [x * x for x in ll]
        assert squared == [100, 400, 900, 1600]

    def test_equality_comparison(self) -> None:
        """Test equality comparison between linked lists."""
        ll1 = LinkedList[int]()
        ll2 = LinkedList[int]()
        
        # Empty lists should be equal
        assert ll1 == ll2
        
        # Add same items
        items = [1, 2, 3]
        for item in items:
            ll1.append(item)
            ll2.append(item)
        
        assert ll1 == ll2
        
        # Different items
        ll2.append(4)
        assert ll1 != ll2
        
        # Different order
        ll3 = LinkedList[int]()
        for item in reversed(items):
            ll3.append(item)
        
        assert ll1 != ll3

    def test_string_representation(self) -> None:
        """Test string representations."""
        ll = LinkedList[str]()
        
        # Empty list
        assert "LinkedList[]" in str(ll)
        
        # List with items
        ll.append("first")
        ll.append("second")
        
        ll_str = str(ll)
        assert "LinkedList[" in ll_str
        assert "first" in ll_str
        assert "second" in ll_str


class TestLinkedListEdgeCases:
    """Test edge cases and error conditions."""

    def test_large_list_operations(self) -> None:
        """Test operations on large lists."""
        ll = LinkedList[int]()
        n = 10000
        
        # Add large number of items
        for i in range(n):
            ll.append(i)
        
        assert len(ll) == n
        
        # Random access
        assert ll.get(n // 2) == n // 2
        assert ll.get(n - 1) == n - 1
        
        # Remove from middle
        ll.remove(n // 2)
        assert len(ll) == n - 1

    def test_mixed_operations(self) -> None:
        """Test mixed append, prepend, insert operations."""
        ll = LinkedList[str]()
        
        ll.append("middle")
        ll.prepend("start")
        ll.append("end")
        ll.insert(1, "between")
        
        expected = ["start", "between", "middle", "end"]
        assert ll.to_list() == expected

    def test_boundary_conditions(self) -> None:
        """Test boundary conditions."""
        ll = LinkedList[int]()
        
        # Insert at index 0 in empty list
        ll.insert(0, 42)
        assert ll.get(0) == 42
        
        # Insert at last position
        ll.insert(1, 43)
        assert ll.get(1) == 43
        
        # Remove last item
        ll.remove(1)
        assert len(ll) == 1
        
        # Remove only item
        ll.remove(0)
        assert ll.is_empty()

    def test_type_safety(self) -> None:
        """Test type safety with different types."""
        # Custom object linked list
        class TestObject:
            def __init__(self, value: int):
                self.value = value
            
            def __eq__(self, other):
                return isinstance(other, TestObject) and self.value == other.value
            
            def __repr__(self):
                return f"TestObject({self.value})"
        
        ll = LinkedList[TestObject]()
        obj1 = TestObject(1)
        obj2 = TestObject(2)
        
        ll.append(obj1)
        ll.append(obj2)
        
        assert ll.get(0) == obj1
        assert ll.get(1) == obj2
        assert ll.find(obj1) == 0
        assert ll.remove_value(obj1) == True


class TestLinkedListPerformance:
    """Test performance characteristics."""

    def test_append_performance(self) -> None:
        """Test append performance (should be O(1))."""
        ll = LinkedList[int]()
        
        start_time = time.time()
        for i in range(10000):
            ll.append(i)
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 0.1
        assert len(ll) == 10000

    def test_prepend_performance(self) -> None:
        """Test prepend performance (should be O(1))."""
        ll = LinkedList[int]()
        
        start_time = time.time()
        for i in range(10000):
            ll.prepend(i)
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 0.1
        assert len(ll) == 10000

    def test_random_access_performance(self) -> None:
        """Test random access performance (O(n) but should be reasonable)."""
        ll = LinkedList[int]()
        n = 1000
        
        for i in range(n):
            ll.append(i)
        
        start_time = time.time()
        for i in range(100):  # Access 100 random positions
            ll.get(i * (n // 100))
        end_time = time.time()
        
        # Should complete reasonably quickly
        assert end_time - start_time < 0.1


if __name__ == "__main__":
    pytest.main([__file__])

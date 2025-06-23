"""
Test suite for the BinaryHeap data structure.

This module provides testing of the BinaryHeap implementation including
edge cases, performance validation, and error handling.
"""

import pytest
from typing import List, Callable
import time
import random

from pyalgos.heaps.binary_heap import BinaryHeap
from pyalgos.exceptions import EmptyStructureError, InvalidOperationError


class TestBinaryHeapBasicOperations:
    """Test basic binary heap operations and functionality."""

    def test_empty_heap_creation(self) -> None:
        """Test creating an empty heap."""
        heap = BinaryHeap[int]()
        assert len(heap) == 0
        assert heap.is_empty()

    def test_min_heap_behavior(self) -> None:
        """Test min-heap behavior (default)."""
        heap = BinaryHeap[int]()
        
        items = [5, 2, 8, 1, 9, 3]
        for item in items:
            heap.insert(item)
        
        assert len(heap) == 6
        assert heap.peek_min() == 1  # Minimum element
        
        # Extract all elements - should be in ascending order
        result = []
        while not heap.is_empty():
            result.append(heap.extract_min())
        
        assert result == sorted(items)

    def test_max_heap_behavior(self) -> None:
        """Test max-heap behavior with custom comparator."""
        # Create max heap with custom comparator
        heap = BinaryHeap[int](compare_func=lambda x, y: x > y)
        
        items = [5, 2, 8, 1, 9, 3]
        for item in items:
            heap.insert(item)
        
        assert len(heap) == 6
        assert heap.peek_min() == 9  # Maximum element (in max-heap context)
        
        # Extract all elements - should be in descending order
        result = []
        while not heap.is_empty():
            result.append(heap.extract_min())
        
        assert result == sorted(items, reverse=True)

    def test_custom_comparator(self) -> None:
        """Test heap with custom comparator."""
        # Heap that compares strings by length
        def compare_by_length(a: str, b: str) -> bool:
            return len(a) < len(b)
        
        heap = BinaryHeap[str](compare_func=compare_by_length)
        
        words = ["hello", "a", "world", "hi", "programming"]
        for word in words:
            heap.insert(word)
        
        # Should extract shortest strings first
        result = []
        while not heap.is_empty():
            result.append(heap.extract_min())
        
        # Verify sorted by length (first few should be shorter)
        assert len(result[0]) <= len(result[1])

    def test_insert_and_extract_operations(self) -> None:
        """Test insert and extract operations."""
        heap = BinaryHeap[int]()
        
        # Insert single item
        heap.insert(42)
        assert heap.peek_min() == 42
        assert len(heap) == 1
        
        # Extract single item
        result = heap.extract_min()
        assert result == 42
        assert heap.is_empty()
        
        # Insert multiple items
        items = [3, 1, 4, 1, 5, 9]
        for item in items:
            heap.insert(item)
        
        # Extract all items
        results = []
        while not heap.is_empty():
            results.append(heap.extract_min())
        
        assert results == sorted(items)

    def test_peek_operations(self) -> None:
        """Test peek operations."""
        heap = BinaryHeap[int]()
        
        # Peek on empty heap
        with pytest.raises(EmptyStructureError):
            heap.peek_min()
        
        # Peek with items
        heap.insert(10)
        heap.insert(5)
        heap.insert(15)
        
        assert heap.peek_min() == 5  # Should be minimum
        assert len(heap) == 3  # Peek shouldn't remove items
        
        # Peek after extract
        heap.extract_min()
        assert heap.peek_min() == 10

    def test_extract_empty_heap(self) -> None:
        """Test extract on empty heap raises appropriate error."""
        heap = BinaryHeap[int]()
        
        with pytest.raises(EmptyStructureError):
            heap.extract_min()

    def test_heapify_from_list(self) -> None:
        """Test creating heap from existing list."""
        items = [4, 10, 3, 5, 1]
        heap = BinaryHeap.from_list(items)
        
        assert len(heap) == 5
        assert heap.peek_min() == 1  # Minimum element
        
        # Verify heap property
        result = []
        while not heap.is_empty():
            result.append(heap.extract_min())
        
        assert result == sorted(items)

    def test_string_representation(self) -> None:
        """Test string representations."""
        heap = BinaryHeap[int]()
        
        # Empty heap
        assert "BinaryHeap[]" in str(heap)
        
        # Heap with items
        items = [3, 1, 4]
        for item in items:
            heap.insert(item)
        
        heap_str = str(heap)
        assert "BinaryHeap[" in heap_str
        # Should show heap array representation
        assert "1" in heap_str  # Min element should be first


class TestBinaryHeapAdvancedOperations:
    """Test advanced heap operations."""

    def test_decrease_key(self) -> None:
        """Test decrease key operation."""
        heap = BinaryHeap.from_list([5, 10, 15, 20])
        
        # Decrease key at index 2 
        heap.decrease_key(2, 3)
        assert heap.peek_min() == 3  # Should be new minimum

    def test_merge_heaps(self) -> None:
        """Test merging two heaps."""
        heap1 = BinaryHeap[int]()
        heap2 = BinaryHeap[int]()
        
        # Add items to both heaps
        for i in [1, 3, 5]:
            heap1.insert(i)
        for i in [2, 4, 6]:
            heap2.insert(i)
        
        # Merge heap2 into heap1
        heap1.merge(heap2)
        
        assert len(heap1) == 6
        assert heap2.is_empty()  # heap2 should be empty after merge
        
        # Extract all items - should be sorted
        result = []
        while not heap1.is_empty():
            result.append(heap1.extract_min())
        
        assert result == [1, 2, 3, 4, 5, 6]

    def test_clear_operation(self) -> None:
        """Test clearing the heap."""
        heap = BinaryHeap[int]()
        
        # Add items
        for i in range(10):
            heap.insert(i)
        
        # Clear and verify
        heap.clear()
        assert len(heap) == 0
        assert heap.is_empty()
        
        # Should be able to add items after clear
        heap.insert(42)
        assert heap.peek_min() == 42


class TestBinaryHeapEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_element_operations(self) -> None:
        """Test operations with single element."""
        heap = BinaryHeap[int]()
        heap.insert(42)
        
        assert heap.peek_min() == 42
        assert len(heap) == 1
        
        result = heap.extract_min()
        assert result == 42
        assert heap.is_empty()

    def test_duplicate_elements(self) -> None:
        """Test heap with duplicate elements."""
        heap = BinaryHeap[int]()
        items = [5, 5, 5, 3, 3, 7, 7, 7]
        
        for item in items:
            heap.insert(item)
        
        # Should handle duplicates correctly
        result = []
        while not heap.is_empty():
            result.append(heap.extract_min())
        
        assert result == sorted(items)

    def test_large_heap_operations(self) -> None:
        """Test operations on large heaps."""
        heap = BinaryHeap[int]()
        n = 1000  # Reduced for reasonable test time
        
        # Add large number of items
        items = list(range(n))
        random.shuffle(items)
        
        for item in items:
            heap.insert(item)
        
        assert len(heap) == n
        assert heap.peek_min() == 0  # Minimum
        
        # Extract first few items to verify
        for expected in range(10):
            assert heap.extract_min() == expected

    def test_heap_with_custom_objects(self) -> None:
        """Test heap with custom objects."""
        class Task:
            def __init__(self, priority: int, name: str):
                self.priority = priority
                self.name = name
            
            def __eq__(self, other):
                return isinstance(other, Task) and self.priority == other.priority
            
            def __repr__(self):
                return f"Task({self.priority}, '{self.name}')"
        
        # Priority queue - lower priority number = higher priority
        def compare_tasks(a: Task, b: Task) -> bool:
            return a.priority < b.priority
        
        heap = BinaryHeap[Task](compare_func=compare_tasks)
        
        tasks = [
            Task(3, "Low priority"),
            Task(1, "High priority"),
            Task(2, "Medium priority"),
            Task(1, "Another high priority")
        ]
        
        for task in tasks:
            heap.insert(task)
        
        # Should extract highest priority tasks first
        result = heap.extract_min()
        assert result.priority == 1
        
        result = heap.extract_min()
        assert result.priority == 1
        
        result = heap.extract_min()
        assert result.priority == 2

    def test_error_conditions(self) -> None:
        """Test various error conditions."""
        heap = BinaryHeap[int]()
        
        # Operations on empty heap
        with pytest.raises(EmptyStructureError):
            heap.peek_min()
        
        with pytest.raises(EmptyStructureError):
            heap.extract_min()
        
        # Invalid decrease_key operations
        heap.insert(10)
        with pytest.raises(InvalidOperationError):
            heap.decrease_key(0, 5)  # Invalid index
        
        with pytest.raises(InvalidOperationError):
            heap.decrease_key(5, 5)  # Index out of bounds
        
        with pytest.raises(InvalidOperationError):
            heap.decrease_key(1, 15)  # New value not smaller

    def test_invalid_initialization(self) -> None:
        """Test invalid initialization parameters."""
        with pytest.raises(InvalidOperationError):
            BinaryHeap[int](initial_capacity=0)
        
        with pytest.raises(InvalidOperationError):
            BinaryHeap[int](initial_capacity=-5)


class TestBinaryHeapPerformance:
    """Test performance characteristics."""

    def test_insert_performance(self) -> None:
        """Test insert performance (should be O(log n))."""
        heap = BinaryHeap[int]()
        n = 1000  # Reasonable size for testing
        
        start_time = time.time()
        for i in range(n):
            heap.insert(random.randint(1, 1000))
        end_time = time.time()
        
        # Should complete reasonably quickly
        assert end_time - start_time < 1.0
        assert len(heap) == n

    def test_extract_performance(self) -> None:
        """Test extract performance (should be O(log n))."""
        n = 1000
        items = [random.randint(1, 1000) for _ in range(n)]
        heap = BinaryHeap.from_list(items)
        
        start_time = time.time()
        for i in range(n):
            heap.extract_min()
        end_time = time.time()
        
        # Should complete reasonably quickly
        assert end_time - start_time < 1.0
        assert heap.is_empty()

    def test_heapify_performance(self) -> None:
        """Test heapify performance (should be O(n))."""
        n = 1000
        items = [random.randint(1, 1000) for _ in range(n)]
        
        start_time = time.time()
        heap = BinaryHeap.from_list(items)
        end_time = time.time()
        
        # Heapify should be reasonably fast
        assert end_time - start_time < 0.5
        assert len(heap) == n
        assert not heap.is_empty()


class TestBinaryHeapUtilities:
    """Test utility methods and features."""

    def test_to_list_conversion(self) -> None:
        """Test conversion to list."""
        items = [3, 1, 4, 1, 5, 9]
        heap = BinaryHeap.from_list(items)
        
        # to_list should return heap array representation
        heap_list = heap.to_list()
        assert len(heap_list) == len(items)
        assert heap_list[0] == 1  # Root should be minimum

    def test_contains_operation(self) -> None:
        """Test membership testing."""
        items = [1, 2, 3, 4, 5]
        heap = BinaryHeap.from_list(items)
        
        for item in items:
            assert item in heap
        
        assert 6 not in heap
        assert 0 not in heap

    def test_iteration(self) -> None:
        """Test iteration over heap."""
        items = [3, 1, 4, 1, 5, 9]
        heap = BinaryHeap.from_list(items)
        
        # Iteration should visit all elements
        iterated_items = []
        for item in heap:
            iterated_items.append(item)
        
        assert len(iterated_items) == len(items)
        # All original items should be present
        for item in items:
            assert item in iterated_items


if __name__ == "__main__":
    pytest.main([__file__])

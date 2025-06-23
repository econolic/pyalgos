"""
Test suite for the Queue data structure.

This module provides thorough testing of the Queue implementation including
edge cases, performance validation, and error handling.
"""

import pytest
from typing import List
import time

from pyalgos.linear.queue import Queue
from pyalgos.exceptions import EmptyStructureError, InvalidOperationError


class TestQueueBasicOperations:
    """Test basic queue operations and functionality."""

    def test_empty_queue_creation(self) -> None:
        """Test creating an empty queue."""
        queue = Queue[int]()
        assert len(queue) == 0
        assert queue.is_empty()
        assert queue.get_capacity() >= 8  # At least initial capacity

    def test_custom_initial_capacity(self) -> None:
        """Test creating a queue with custom initial capacity."""
        queue = Queue[str](initial_capacity=16)
        assert len(queue) == 0
        assert queue.is_empty()
        assert queue.get_capacity() >= 16

    def test_invalid_initial_capacity(self) -> None:
        """Test that invalid initial capacity raises an error."""
        with pytest.raises(InvalidOperationError):
            Queue[int](initial_capacity=0)
            
        with pytest.raises(InvalidOperationError):
            Queue[int](initial_capacity=-5)

    def test_single_enqueue_dequeue(self) -> None:
        """Test basic enqueue and dequeue operations."""
        queue = Queue[int]()
        queue.enqueue(42)
        
        assert len(queue) == 1
        assert not queue.is_empty()
        assert queue.peek() == 42
        
        result = queue.dequeue()
        assert result == 42
        assert len(queue) == 0
        assert queue.is_empty()

    def test_multiple_enqueue_dequeue(self) -> None:
        """Test multiple enqueue and dequeue operations."""
        queue = Queue[str]()
        items = ["first", "second", "third"]
        
        # Enqueue all items
        for item in items:
            queue.enqueue(item)
        
        assert len(queue) == 3
        assert queue.peek() == "first"
        
        # Dequeue all items in FIFO order
        for expected_item in items:
            assert queue.dequeue() == expected_item
        
        assert queue.is_empty()

    def test_peek_operations(self) -> None:
        """Test peek operations."""
        queue = Queue[int]()
        
        # Test peek on empty queue
        with pytest.raises(EmptyStructureError):
            queue.peek()
        
        # Test peek with items
        queue.enqueue(10)
        queue.enqueue(20)
        
        assert queue.peek() == 10  # Should always return first item
        assert len(queue) == 2  # Peek shouldn't remove items
        
        queue.dequeue()
        assert queue.peek() == 20

    def test_dequeue_empty_queue(self) -> None:
        """Test dequeue on empty queue raises appropriate error."""
        queue = Queue[int]()
        
        with pytest.raises(EmptyStructureError):
            queue.dequeue()

    def test_fifo_order(self) -> None:
        """Test that queue maintains FIFO order."""
        queue = Queue[int]()
        items = list(range(100))
        
        # Enqueue all items
        for item in items:
            queue.enqueue(item)
        
        # Dequeue and verify order
        results = []
        while not queue.is_empty():
            results.append(queue.dequeue())
        
        assert results == items  # Should be same order

    def test_circular_buffer_wraparound(self) -> None:
        """Test circular buffer wraparound functionality."""
        queue = Queue[int](initial_capacity=4)
        
        # Fill queue
        for i in range(4):
            queue.enqueue(i)
        
        # Remove some items
        assert queue.dequeue() == 0
        assert queue.dequeue() == 1
        
        # Add more items to test wraparound
        queue.enqueue(4)
        queue.enqueue(5)
        
        # Verify correct order
        assert queue.dequeue() == 2
        assert queue.dequeue() == 3
        assert queue.dequeue() == 4
        assert queue.dequeue() == 5
        assert queue.is_empty()

    def test_dynamic_resizing(self) -> None:
        """Test dynamic resizing when queue becomes full."""
        queue = Queue[int](initial_capacity=4)
        initial_capacity = queue.get_capacity()
        
        # Fill beyond initial capacity
        for i in range(10):
            queue.enqueue(i)
        
        assert len(queue) == 10
        assert queue.get_capacity() > initial_capacity
        
        # Verify all items are preserved
        for i in range(10):
            assert queue.dequeue() == i

    def test_repr_and_str(self) -> None:
        """Test string representations."""
        queue = Queue[str]()
        
        # Empty queue
        assert "Queue[]" in str(queue)
        
        # Queue with items
        queue.enqueue("first")
        queue.enqueue("second")
        
        queue_str = str(queue)
        assert "Queue[" in queue_str
        assert "first" in queue_str
        assert "second" in queue_str


class TestQueueEdgeCases:
    """Test edge cases and error conditions."""

    def test_large_queue_operations(self) -> None:
        """Test operations on large queues."""
        queue = Queue[int]()
        n = 10000
        
        # Add large number of items
        for i in range(n):
            queue.enqueue(i)
        
        assert len(queue) == n
        
        # Remove all items
        for i in range(n):
            assert queue.dequeue() == i
        
        assert queue.is_empty()

    def test_alternating_operations(self) -> None:
        """Test alternating enqueue/dequeue operations."""
        queue = Queue[int]()
        
        for i in range(1000):
            queue.enqueue(i)
            if i % 2 == 1:  # Dequeue every other iteration
                queue.dequeue()
        
        # Should have approximately half the items
        assert 400 <= len(queue) <= 600

    def test_clear_operation(self) -> None:
        """Test clearing the queue."""
        queue = Queue[int]()
        
        # Add items
        for i in range(10):
            queue.enqueue(i)
        
        # Clear and verify
        queue.clear()
        assert len(queue) == 0
        assert queue.is_empty()
        
        # Should be able to add items after clear
        queue.enqueue(42)
        assert queue.dequeue() == 42

    def test_type_safety(self) -> None:
        """Test type safety with different types."""
        # String queue
        str_queue = Queue[str]()
        str_queue.enqueue("hello")
        str_queue.enqueue("world")
        
        assert str_queue.dequeue() == "hello"
        assert str_queue.dequeue() == "world"
        
        # Custom object queue
        class TestObject:
            def __init__(self, value: int):
                self.value = value
            
            def __eq__(self, other):
                return isinstance(other, TestObject) and self.value == other.value
        
        obj_queue = Queue[TestObject]()
        obj1 = TestObject(1)
        obj2 = TestObject(2)
        
        obj_queue.enqueue(obj1)
        obj_queue.enqueue(obj2)
        
        assert obj_queue.dequeue() == obj1
        assert obj_queue.dequeue() == obj2


class TestQueuePerformance:
    """Test performance characteristics."""

    def test_operation_time_complexity(self) -> None:
        """Test that basic operations maintain expected time complexity."""
        queue = Queue[int]()
        
        # Test enqueue performance
        start_time = time.time()
        for i in range(10000):
            queue.enqueue(i)
        enqueue_time = time.time() - start_time
        
        # Test dequeue performance
        start_time = time.time()
        for i in range(10000):
            queue.dequeue()
        dequeue_time = time.time() - start_time
        
        # Operations should be very fast (O(1) amortized)
        assert enqueue_time < 0.1  # Should complete in well under 100ms
        assert dequeue_time < 0.1

    def test_memory_efficiency(self) -> None:
        """Test memory usage patterns."""
        queue = Queue[int]()
        
        # Fill queue
        for i in range(1000):
            queue.enqueue(i)
        
        capacity_after_fill = queue.get_capacity()
        
        # Empty queue partially
        for i in range(500):
            queue.dequeue()
        
        # Capacity shouldn't grow unnecessarily
        assert queue.get_capacity() == capacity_after_fill
        
        # Should still work correctly
        for i in range(500):
            assert queue.dequeue() == i + 500


if __name__ == "__main__":
    pytest.main([__file__])

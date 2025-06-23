"""
Test suite for the Stack data structure.

This module provides thorough testing of the Stack implementation including
edge cases, performance validation, and error handling. The tests follow
the patterns established in advanced algorithm testing frameworks.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from pyalgos.exceptions import EmptyStructureError, InvalidOperationError
from pyalgos.linear.stack import Stack


class TestStackBasicOperations:
    """Test basic stack operations and functionality."""

    def test_empty_stack_creation(self) -> None:
        """Test creating an empty stack."""
        stack = Stack[int]()
        assert len(stack) == 0
        assert stack.is_empty()
        assert stack.get_capacity() == 8  # Default initial capacity

    def test_custom_initial_capacity(self) -> None:
        """Test creating a stack with custom initial capacity."""
        stack = Stack[str](initial_capacity=16)
        assert len(stack) == 0
        assert stack.is_empty()
        assert stack.get_capacity() == 16

    def test_invalid_initial_capacity(self) -> None:
        """Test that invalid initial capacity raises an error."""
        with pytest.raises(InvalidOperationError):
            Stack[int](initial_capacity=0)
            
        with pytest.raises(InvalidOperationError):
            Stack[int](initial_capacity=-5)

    def test_single_push_pop(self) -> None:
        """Test basic push and pop operations."""
        stack = Stack[int]()
        stack.push(42)
        
        assert len(stack) == 1
        assert not stack.is_empty()
        assert stack.peek() == 42
        
        popped = stack.pop()
        assert popped == 42
        assert len(stack) == 0
        assert stack.is_empty()

    def test_multiple_push_pop(self) -> None:
        """Test multiple push and pop operations."""
        stack = Stack[str]()
        items = ["first", "second", "third", "fourth"]
        
        # Push items
        for item in items:
            stack.push(item)
            
        assert len(stack) == len(items)
        
        # Pop items in LIFO order
        for expected_item in reversed(items):
            assert stack.peek() == expected_item
            popped = stack.pop()
            assert popped == expected_item

    def test_peek_operations(self) -> None:
        """Test peek functionality."""
        stack = Stack[float]()
        
        # Test peek on different stack sizes
        stack.push(1.1)
        assert stack.peek() == 1.1
        assert len(stack) == 1  # Peek shouldn't modify stack
        
        stack.push(2.2)
        assert stack.peek() == 2.2
        assert len(stack) == 2
        
        stack.push(3.3)
        assert stack.peek() == 3.3
        assert len(stack) == 3


class TestStackErrorHandling:
    """Test error handling and edge cases."""

    def test_pop_empty_stack(self) -> None:
        """Test that popping from empty stack raises error."""
        stack = Stack[int]()
        with pytest.raises(EmptyStructureError):
            stack.pop()

    def test_peek_empty_stack(self) -> None:
        """Test that peeking empty stack raises error."""
        stack = Stack[str]()
        with pytest.raises(EmptyStructureError):
            stack.peek()

    def test_push_none_value(self) -> None:
        """Test that pushing None raises an error."""
        stack = Stack[Any]()
        with pytest.raises(InvalidOperationError):
            stack.push(None)

    def test_operations_after_clear(self) -> None:
        """Test that operations work correctly after clearing."""
        stack = Stack[int]()
        
        # Add some items
        for i in range(5):
            stack.push(i)
            
        # Clear and verify
        stack.clear()
        assert len(stack) == 0
        assert stack.is_empty()
        assert stack.get_capacity() == 8  # Reset to initial capacity
        
        # Test operations still work
        stack.push(100)
        assert stack.peek() == 100
        assert len(stack) == 1


class TestStackResizing:
    """Test automatic resizing functionality."""

    def test_automatic_growth(self) -> None:
        """Test that stack grows automatically when capacity is exceeded."""
        stack = Stack[int](initial_capacity=4)
        assert stack.get_capacity() == 4
        
        # Fill to capacity
        for i in range(4):
            stack.push(i)
        assert stack.get_capacity() == 4
        
        # Add one more to trigger resize
        stack.push(4)
        assert stack.get_capacity() == 8  # Should double
        assert len(stack) == 5
        
        # Verify all elements are still accessible
        expected = [4, 3, 2, 1, 0]
        for expected_val in expected:
            assert stack.pop() == expected_val

    def test_automatic_shrinking(self) -> None:
        """Test that stack shrinks when utilization is low."""
        stack = Stack[int](initial_capacity=16)
        
        # Fill stack to trigger potential shrinking scenario
        for i in range(16):
            stack.push(i)
        assert stack.get_capacity() == 16
        
        # Remove most elements to trigger shrinking
        for _ in range(12):  # Leave 4 elements, 25% utilization
            stack.pop()
            
        # Should shrink after next pop (below 25% threshold)
        stack.pop()  # Now 3 elements in capacity 16 = 18.75%
        assert stack.get_capacity() == 8  # Should have shrunk
        assert len(stack) == 3

    def test_minimum_capacity_maintained(self) -> None:
        """Test that stack maintains minimum capacity."""
        stack = Stack[int](initial_capacity=4)
        
        # Add and remove elements
        stack.push(1)
        stack.push(2)
        stack.pop()
        stack.pop()
        
        # Should not shrink below minimum capacity
        assert stack.get_capacity() >= 4

    def test_load_factor_calculation(self) -> None:
        """Test load factor calculation."""
        stack = Stack[int](initial_capacity=8)
        
        assert stack.get_load_factor() == 0.0  # Empty stack
        
        stack.push(1)
        assert stack.get_load_factor() == 0.125  # 1/8
        
        for i in range(3):
            stack.push(i)
        assert stack.get_load_factor() == 0.5  # 4/8


class TestStackIteration:
    """Test iteration and container protocol functionality."""

    def test_iteration_order(self) -> None:
        """Test that iteration follows LIFO order."""
        stack = Stack[str]()
        items = ["bottom", "middle", "top"]
        
        for item in items:
            stack.push(item)
            
        # Iteration should be top to bottom (LIFO)
        result = list(stack)
        expected = ["top", "middle", "bottom"]
        assert result == expected

    def test_empty_stack_iteration(self) -> None:
        """Test iteration over empty stack."""
        stack = Stack[int]()
        result = list(stack)
        assert result == []

    def test_contains_operation(self) -> None:
        """Test the 'in' operator."""
        stack = Stack[int]()
        
        # Empty stack
        assert 1 not in stack
        
        # Add items
        for i in range(5):
            stack.push(i)
            
        # Test membership
        for i in range(5):
            assert i in stack
            
        assert 10 not in stack
        assert -1 not in stack

    def test_string_representations(self) -> None:
        """Test string representation methods."""
        stack = Stack[int]()
        
        # Empty stack
        assert str(stack) == "[]"
        assert "Stack" in repr(stack)
        assert "size=0" in repr(stack)
        
        # Non-empty stack
        stack.push(1)
        stack.push(2)
        stack.push(3)
        
        assert str(stack) == "[3, 2, 1]"  # LIFO order
        assert "Stack" in repr(stack)
        assert "size=3" in repr(stack)


class TestStackPerformance:
    """Test performance characteristics and complexity validation."""

    def test_push_pop_performance(self) -> None:
        """Test that push/pop operations maintain O(1) amortized complexity."""
        stack = Stack[int]()
        n = 10000
        
        # Measure push operations
        start_time = time.perf_counter()
        for i in range(n):
            stack.push(i)
        push_time = time.perf_counter() - start_time
        
        assert len(stack) == n
        
        # Measure pop operations
        start_time = time.perf_counter()
        for _ in range(n):
            stack.pop()
        pop_time = time.perf_counter() - start_time
        
        assert stack.is_empty()
        
        # Performance should be reasonable (these are loose bounds)
        assert push_time < 1.0  # Should complete well under 1 second
        assert pop_time < 1.0
        
        print(f"Push {n} items: {push_time:.4f}s")
        print(f"Pop {n} items: {pop_time:.4f}s")

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency through load factor monitoring."""
        stack = Stack[int](initial_capacity=8)
        
        # Test growth efficiency
        for i in range(20):
            stack.push(i)
            load_factor = stack.get_load_factor()
            # Load factor should never be too low after growth
            if stack.get_capacity() > 8:  # After first resize
                assert load_factor >= 0.25
                
        # Test shrink efficiency
        while len(stack) > 2:
            stack.pop()
            load_factor = stack.get_load_factor()
            # After shrinking, load factor should be reasonable
            assert load_factor <= 1.0


class TestStackTypeHandling:
    """Test type safety and generic type handling."""

    def test_different_types(self) -> None:
        """Test stack with different data types."""
        # Integer stack
        int_stack = Stack[int]()
        int_stack.push(42)
        assert int_stack.pop() == 42
        
        # String stack
        str_stack = Stack[str]()
        str_stack.push("hello")
        assert str_stack.pop() == "hello"
        
        # Float stack
        float_stack = Stack[float]()
        float_stack.push(3.14)
        assert float_stack.pop() == 3.14
        
        # Boolean stack
        bool_stack = Stack[bool]()
        bool_stack.push(True)
        bool_stack.push(False)
        assert bool_stack.pop() is False
        assert bool_stack.pop() is True

    def test_complex_objects(self) -> None:
        """Test stack with complex objects."""
        class TestObject:
            def __init__(self, value: int) -> None:
                self.value = value
                
            def __eq__(self, other: object) -> bool:
                return isinstance(other, TestObject) and self.value == other.value
        
        stack = Stack[TestObject]()
        obj1 = TestObject(1)
        obj2 = TestObject(2)
        
        stack.push(obj1)
        stack.push(obj2)
        
        assert stack.pop() == obj2
        assert stack.pop() == obj1


class TestStackIntegration:
    """Integration tests combining multiple operations."""

    def test_comprehensive_workflow(self) -> None:
        """Test a comprehensive workflow combining all operations."""
        stack = Stack[str]()
        
        # Initial state
        assert stack.is_empty()
        assert len(stack) == 0
        
        # Build up stack
        words = ["apple", "banana", "cherry", "date", "elderberry"]
        for word in words:
            stack.push(word)
            assert stack.peek() == word
            
        assert len(stack) == len(words)
        assert not stack.is_empty()
        
        # Test iteration without modification
        stack_contents = list(stack)
        assert len(stack_contents) == len(words)
        assert stack_contents == list(reversed(words))  # LIFO order
        
        # Test membership
        for word in words:
            assert word in stack
            
        # Partial removal
        for _ in range(2):
            stack.pop()
            
        assert len(stack) == 3
        remaining_words = words[:3]  # First 3 words
        
        # Verify remaining contents
        for word in reversed(remaining_words):
            assert stack.pop() == word
            
        assert stack.is_empty()

    def test_stress_operations(self) -> None:
        """Stress test with many operations."""
        stack = Stack[int]()
        operations = 1000
        
        # Mixed push/pop operations
        for i in range(operations):
            if i % 3 == 0 and not stack.is_empty():
                # Pop occasionally
                stack.pop()
            else:
                # Push most of the time
                stack.push(i)
                
        # Clean up remaining items
        final_size = len(stack)
        for _ in range(final_size):
            stack.pop()
            
        assert stack.is_empty()


def run_comprehensive_tests() -> None:
    """
    Run all tests with detailed output.
    
    This function mimics the testing approach from advanced algorithm
    implementations, providing comprehensive validation of the Stack.
    """
    print("=" * 60)
    print("COMPREHENSIVE STACK TESTING")
    print("=" * 60)
    
    # Use pytest to run all tests
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_comprehensive_tests()

"""
Array-based Stack implementation with type safety and performance analysis.

This module provides a stack data structure that follows LIFO
(Last In, First Out) ordering. The implementation includes robust error handling,
performance measurement capabilities, and testing.

Key Features:
    - Dynamic array-based storage with automatic resizing
    - O(1) amortized push/pop operations
    - Input validation and error handling
    - Memory-efficient with shrinking capability
    - Full type safety with generic type support
    - Performance measurement and analysis tools

Example:
    >>> from pyalgos.linear.stack import Stack
    >>> stack = Stack[int]()
    >>> stack.push(1)
    >>> stack.push(2)
    >>> print(stack.peek())  # 2
    >>> print(stack.pop())   # 2
    >>> print(len(stack))    # 1
"""

from __future__ import annotations

import time
from typing import Any, Generic, Iterator, TypeVar

from pyalgos.exceptions import EmptyStructureError, InvalidOperationError
from pyalgos.types import SizeType

T = TypeVar("T")


def measure_performance(func: Any) -> Any:
    """Decorator to measure function execution time.
    
    This decorator provides performance analysis capabilities similar to
    the patterns used in advanced algorithm implementations. It measures
    the execution time and can be used for complexity analysis.
    
    Args:
        func: The function to be measured.
        
    Returns:
        The wrapped function that includes timing information.
    """
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


class Stack(Generic[T]):
    """
    A generic, array-based stack implementation with automatic resizing.
    
    This stack follows the LIFO (Last In, First Out) principle and provides
    O(1) amortized time complexity for push and pop operations. The implementation
    uses dynamic array resizing to maintain optimal memory usage.
    
    The stack automatically grows when capacity is exceeded and shrinks when
    utilization drops below 25% to maintain memory efficiency.
    
    Type Parameters:
        T: The type of elements stored in the stack.
        
    Attributes:
        _data: Internal array storing stack elements.
        _size: Current number of elements in the stack.
        _capacity: Current capacity of the internal array.
        
    Time Complexities:
        - push(): O(1) amortized
        - pop(): O(1) amortized  
        - peek(): O(1)
        - is_empty(): O(1)
        - __len__(): O(1)
        
    Space Complexity: O(n) where n is the number of elements
    """
    
    # Class constants for array management
    _INITIAL_CAPACITY: int = 8
    _GROWTH_FACTOR: int = 2
    _SHRINK_THRESHOLD: float = 0.25
    _MIN_CAPACITY: int = 4
    
    def __init__(self, initial_capacity: int | None = None) -> None:
        """
        Initialize an empty stack with optional initial capacity.
        
        Args:
            initial_capacity: Optional initial capacity for the internal array.
                            Must be positive if provided. Defaults to 8.
                            
        Raises:
            InvalidOperationError: If initial_capacity is not positive.
            
        Example:
            >>> stack = Stack[str]()
            >>> custom_stack = Stack[int](initial_capacity=16)
        """
        if initial_capacity is not None:
            if not isinstance(initial_capacity, int) or initial_capacity <= 0:
                raise InvalidOperationError(
                    f"Initial capacity must be a positive integer, got {initial_capacity}"
                )
            self._capacity = initial_capacity
        else:
            self._capacity = self._INITIAL_CAPACITY
            
        self._data: list[T | None] = [None] * self._capacity
        self._size: SizeType = 0
        
    def push(self, item: T) -> None:
        """
        Push an item onto the top of the stack.
        
        Automatically resizes the internal array if capacity is exceeded.
        The resize operation doubles the capacity to ensure amortized O(1)
        time complexity.
        
        Args:
            item: The item to push onto the stack.
            
        Time Complexity: O(1) amortized
        Space Complexity: O(1)
        
        Example:
            >>> stack = Stack[int]()
            >>> stack.push(42)
            >>> stack.push(100)
            >>> len(stack)
            2
        """
        if item is None:
            raise InvalidOperationError("Cannot push None onto the stack")
            
        # Resize if necessary
        if self._size >= self._capacity:
            self._resize(self._capacity * self._GROWTH_FACTOR)
            
        self._data[self._size] = item
        self._size += 1
        
    def pop(self) -> T:
        """
        Remove and return the top item from the stack.
        
        Automatically shrinks the internal array if utilization drops below
        the shrink threshold to maintain memory efficiency.
        
        Returns:
            The item that was removed from the top of the stack.
            
        Raises:
            EmptyStructureError: If the stack is empty.
            
        Time Complexity: O(1) amortized
        Space Complexity: O(1)
        
        Example:
            >>> stack = Stack[str]()
            >>> stack.push("hello")
            >>> stack.push("world")
            >>> stack.pop()
            'world'
            >>> stack.pop()
            'hello'
        """
        if self.is_empty():
            raise EmptyStructureError("Stack")
            
        self._size -= 1
        item = self._data[self._size]
        self._data[self._size] = None  # Help garbage collection
        
        # Shrink if utilization is too low
        if (self._size > 0 and 
            self._size <= self._capacity * self._SHRINK_THRESHOLD and
            self._capacity > self._MIN_CAPACITY):
            self._resize(max(self._capacity // self._GROWTH_FACTOR, self._MIN_CAPACITY))
            
        return item  # type: ignore[return-value]
        
    def peek(self) -> T:
        """
        Return the top item without removing it from the stack.
        
        Returns:
            The top item on the stack.
            
        Raises:
            EmptyStructureError: If the stack is empty.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> stack = Stack[int]()
            >>> stack.push(5)
            >>> stack.peek()
            5
            >>> len(stack)  # Item is still in the stack
            1
        """
        if self.is_empty():
            raise EmptyStructureError("Stack")
            
        return self._data[self._size - 1]  # type: ignore[return-value]
        
    def is_empty(self) -> bool:
        """
        Check if the stack is empty.
        
        Returns:
            True if the stack contains no elements, False otherwise.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> stack = Stack[bool]()
            >>> stack.is_empty()
            True
            >>> stack.push(True)
            >>> stack.is_empty()
            False
        """
        return self._size == 0
        
    def clear(self) -> None:
        """
        Remove all elements from the stack and reset to initial capacity.
        
        This operation helps with memory management by resetting the internal
        array to its initial size.
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> stack = Stack[str]()
            >>> stack.push("a")
            >>> stack.push("b")
            >>> len(stack)
            2
            >>> stack.clear()
            >>> len(stack)
            0
        """
        self._data = [None] * self._INITIAL_CAPACITY
        self._size = 0
        self._capacity = self._INITIAL_CAPACITY
        
    def __len__(self) -> int:
        """
        Return the number of elements in the stack.
        
        Returns:
            The number of elements currently in the stack.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> stack = Stack[float]()
            >>> len(stack)
            0
            >>> stack.push(3.14)
            >>> len(stack)
            1
        """
        return self._size
        
    def __contains__(self, item: object) -> bool:
        """
        Check if an item is in the stack.
        
        Note: This operation requires scanning the entire stack and should
        be used sparingly as it has O(n) time complexity.
        
        Args:
            item: The item to search for.
            
        Returns:
            True if the item is found in the stack, False otherwise.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> stack = Stack[int]()
            >>> stack.push(1)
            >>> stack.push(2)
            >>> 2 in stack
            True
            >>> 3 in stack
            False
        """
        for i in range(self._size):
            if self._data[i] == item:
                return True
        return False
        
    def __iter__(self) -> Iterator[T]:
        """
        Provide iteration over stack elements from top to bottom.
        
        Yields elements in LIFO order (top to bottom). The stack is not
        modified during iteration.
        
        Yields:
            Elements from the top of the stack to the bottom.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> stack = Stack[str]()
            >>> stack.push("first")
            >>> stack.push("second")
            >>> stack.push("third")
            >>> list(stack)
            ['third', 'second', 'first']
        """
        # Iterate from top to bottom (LIFO order)
        for i in range(self._size - 1, -1, -1):
            yield self._data[i]  # type: ignore[misc]
            
    def __repr__(self) -> str:
        """
        Return a detailed string representation of the stack.
        
        Returns:
            A string showing the stack contents and metadata.
            
        Example:
            >>> stack = Stack[int]()
            >>> stack.push(1)
            >>> stack.push(2)
            >>> repr(stack)
            'Stack([2, 1], size=2, capacity=8)'
        """
        elements = [str(self._data[i]) for i in range(self._size - 1, -1, -1)]
        return f"Stack([{', '.join(elements)}], size={self._size}, capacity={self._capacity})"
        
    def __str__(self) -> str:
        """
        Return a simple string representation of the stack.
        
        Returns:
            A string showing the stack contents in LIFO order.
            
        Example:
            >>> stack = Stack[str]()
            >>> stack.push("bottom")
            >>> stack.push("top")
            >>> str(stack)
            '[top, bottom]'
        """
        if self.is_empty():
            return "[]"
        elements = [str(self._data[i]) for i in range(self._size - 1, -1, -1)]
        return f"[{', '.join(elements)}]"
        
    def _resize(self, new_capacity: int) -> None:
        """
        Resize the internal array to the specified capacity.
        
        This is a private method used internally for automatic resizing.
        It maintains all existing elements while changing the array size.
        
        Args:
            new_capacity: The new capacity for the internal array.
            
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        old_data = self._data
        self._data = [None] * new_capacity
        self._capacity = new_capacity
        
        # Copy existing elements
        for i in range(self._size):
            self._data[i] = old_data[i]
            
    def get_capacity(self) -> int:
        """
        Get the current capacity of the internal array.
        
        This method is useful for performance analysis and understanding
        the memory usage characteristics of the stack.
        
        Returns:
            The current capacity of the internal array.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> stack = Stack[int]()
            >>> stack.get_capacity()
            8
            >>> for i in range(10):
            ...     stack.push(i)
            >>> stack.get_capacity()  # Automatically resized
            16
        """
        return self._capacity
        
    def get_load_factor(self) -> float:
        """
        Get the current load factor (utilization) of the stack.
        
        The load factor is the ratio of current size to capacity.
        This is useful for performance analysis and memory optimization.
        
        Returns:
            The load factor as a float between 0.0 and 1.0.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> stack = Stack[int]()
            >>> stack.push(1)
            >>> stack.push(2)
            >>> round(stack.get_load_factor(), 2)
            0.25
        """
        return self._size / self._capacity if self._capacity > 0 else 0.0

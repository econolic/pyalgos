"""
Circular array-based Queue implementation with type safety.

This module provides a queue data structure that follows FIFO
(First In, First Out) ordering. The implementation uses a circular array for
optimal memory utilization and provides O(1) enqueue and dequeue operations.

Key Features:
    - Circular array-based storage with automatic resizing
    - O(1) enqueue/dequeue operations
    - Memory-efficient with no data movement during operations
    - Input validation and error handling
    - Full type safety with generic type support
    - Performance measurement and analysis capabilities

Example:
    >>> from pyalgos.linear.queue import Queue
    >>> queue = Queue[str]()
    >>> queue.enqueue("first")
    >>> queue.enqueue("second")
    >>> print(queue.peek())     # "first"
    >>> print(queue.dequeue())  # "first"
    >>> print(len(queue))       # 1
"""

from __future__ import annotations

import time
from typing import Any, Generic, Iterator, TypeVar

from pyalgos.exceptions import EmptyStructureError, InvalidOperationError
from pyalgos.types import SizeType

T = TypeVar("T")


class Queue(Generic[T]):
    """
    A generic, circular array-based queue implementation with automatic resizing.
    
    This queue follows the FIFO (First In, First Out) principle and provides
    O(1) time complexity for enqueue and dequeue operations. The implementation
    uses a circular array to avoid data movement and optimize memory usage.
    
    The queue automatically grows when capacity is exceeded and can optionally
    shrink when utilization drops to maintain memory efficiency.
    
    Type Parameters:
        T: The type of elements stored in the queue.
        
    Attributes:
        _data: Internal circular array storing queue elements.
        _front: Index of the front element (next to dequeue).
        _rear: Index of the next position for enqueue.
        _size: Current number of elements in the queue.
        _capacity: Current capacity of the internal array.
        
    Time Complexities:
        - enqueue(): O(1) amortized
        - dequeue(): O(1)
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
        Initialize an empty queue with optional initial capacity.
        
        Args:
            initial_capacity: Optional initial capacity for the internal array.
                            Must be positive if provided. Defaults to 8.
                            
        Raises:
            InvalidOperationError: If initial_capacity is not positive.
            
        Example:
            >>> queue = Queue[int]()
            >>> custom_queue = Queue[str](initial_capacity=16)
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
        self._front: SizeType = 0
        self._rear: SizeType = 0
        self._size: SizeType = 0
        
    def enqueue(self, item: T) -> None:
        """
        Add an item to the rear of the queue.
        
        Automatically resizes the internal array if capacity is exceeded.
        The resize operation doubles the capacity to ensure amortized O(1)
        time complexity.
        
        Args:
            item: The item to add to the queue.
            
        Time Complexity: O(1) amortized
        Space Complexity: O(1)
        
        Example:
            >>> queue = Queue[str]()
            >>> queue.enqueue("first")
            >>> queue.enqueue("second")
            >>> len(queue)
            2
        """
        if item is None:
            raise InvalidOperationError("Cannot enqueue None into the queue")
            
        # Resize if necessary
        if self._size >= self._capacity:
            self._resize(self._capacity * self._GROWTH_FACTOR)
            
        self._data[self._rear] = item
        self._rear = (self._rear + 1) % self._capacity
        self._size += 1
        
    def dequeue(self) -> T:
        """
        Remove and return the front item from the queue.
        
        Automatically shrinks the internal array if utilization drops below
        the shrink threshold to maintain memory efficiency.
        
        Returns:
            The item that was removed from the front of the queue.
            
        Raises:
            EmptyStructureError: If the queue is empty.
            
        Time Complexity: O(1) amortized
        Space Complexity: O(1)
        
        Example:
            >>> queue = Queue[int]()
            >>> queue.enqueue(1)
            >>> queue.enqueue(2)
            >>> queue.dequeue()
            1
            >>> queue.dequeue()
            2
        """
        if self.is_empty():
            raise EmptyStructureError("Queue")
            
        item = self._data[self._front]
        self._data[self._front] = None  # Help garbage collection
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        
        # Shrink if utilization is too low
        if (self._size > 0 and 
            self._size <= self._capacity * self._SHRINK_THRESHOLD and
            self._capacity > self._MIN_CAPACITY):
            self._resize(max(self._capacity // self._GROWTH_FACTOR, self._MIN_CAPACITY))
            
        return item  # type: ignore[return-value]
        
    def peek(self) -> T:
        """
        Return the front item without removing it from the queue.
        
        Returns:
            The front item in the queue.
            
        Raises:
            EmptyStructureError: If the queue is empty.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> queue = Queue[float]()
            >>> queue.enqueue(3.14)
            >>> queue.peek()
            3.14
            >>> len(queue)  # Item is still in the queue
            1
        """
        if self.is_empty():
            raise EmptyStructureError("Queue")
            
        return self._data[self._front]  # type: ignore[return-value]
        
    def is_empty(self) -> bool:
        """
        Check if the queue is empty.
        
        Returns:
            True if the queue contains no elements, False otherwise.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> queue = Queue[bool]()
            >>> queue.is_empty()
            True
            >>> queue.enqueue(False)
            >>> queue.is_empty()
            False
        """
        return self._size == 0
        
    def clear(self) -> None:
        """
        Remove all elements from the queue and reset to initial capacity.
        
        This operation helps with memory management by resetting the internal
        array to its initial size and indices to zero.
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> queue = Queue[str]()
            >>> queue.enqueue("a")
            >>> queue.enqueue("b")
            >>> len(queue)
            2
            >>> queue.clear()
            >>> len(queue)
            0
        """
        self._data = [None] * self._INITIAL_CAPACITY
        self._front = 0
        self._rear = 0
        self._size = 0
        self._capacity = self._INITIAL_CAPACITY
        
    def __len__(self) -> int:
        """
        Return the number of elements in the queue.
        
        Returns:
            The number of elements currently in the queue.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> queue = Queue[int]()
            >>> len(queue)
            0
            >>> queue.enqueue(42)
            >>> len(queue)
            1
        """
        return self._size
        
    def __contains__(self, item: object) -> bool:
        """
        Check if an item is in the queue.
        
        Note: This operation requires scanning the queue and should
        be used sparingly as it has O(n) time complexity.
        
        Args:
            item: The item to search for.
            
        Returns:
            True if the item is found in the queue, False otherwise.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> queue = Queue[str]()
            >>> queue.enqueue("hello")
            >>> queue.enqueue("world")
            >>> "hello" in queue
            True
            >>> "python" in queue
            False
        """
        if self.is_empty():
            return False
            
        # Search through circular array
        current = self._front
        for _ in range(self._size):
            if self._data[current] == item:
                return True
            current = (current + 1) % self._capacity
        return False
        
    def __iter__(self) -> Iterator[T]:
        """
        Provide iteration over queue elements from front to rear.
        
        Yields elements in FIFO order (front to rear). The queue is not
        modified during iteration.
        
        Yields:
            Elements from the front of the queue to the rear.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> queue = Queue[int]()
            >>> queue.enqueue(1)
            >>> queue.enqueue(2)
            >>> queue.enqueue(3)
            >>> list(queue)
            [1, 2, 3]
        """
        if self.is_empty():
            return
            
        # Iterate from front to rear (FIFO order)
        current = self._front
        for _ in range(self._size):
            yield self._data[current]  # type: ignore[misc]
            current = (current + 1) % self._capacity
            
    def __repr__(self) -> str:
        """
        Return a detailed string representation of the queue.
        
        Returns:
            A string showing the queue contents and metadata.
            
        Example:
            >>> queue = Queue[str]()
            >>> queue.enqueue("first")
            >>> queue.enqueue("second")
            >>> repr(queue)
            'Queue([first, second], size=2, capacity=8)'
        """
        elements = [str(item) for item in self]
        return f"Queue([{', '.join(elements)}], size={self._size}, capacity={self._capacity})"
        
    def __str__(self) -> str:
        """
        Return a simple string representation of the queue.
          Returns:
            A string showing the queue contents in FIFO order.
            
        Example:
            >>> queue = Queue[int]()
            >>> queue.enqueue(10)
            >>> queue.enqueue(20)
            >>> str(queue)
            '[10, 20]'
        """
        if self.is_empty():
            return "Queue[]"
        elements = [str(item) for item in self]
        return f"Queue[{', '.join(elements)}]"
        
    def _resize(self, new_capacity: int) -> None:
        """
        Resize the internal array to the specified capacity.
        
        This method handles the circular array reorganization, ensuring
        that elements maintain their correct order after resizing.
        
        Args:
            new_capacity: The new capacity for the internal array.
            
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        old_data = self._data
        self._data = [None] * new_capacity
        
        # Copy elements in correct order to new array
        if not self.is_empty():
            current = self._front
            for i in range(self._size):
                self._data[i] = old_data[current]
                current = (current + 1) % self._capacity
                
        # Reset indices
        self._front = 0
        self._rear = self._size
        self._capacity = new_capacity
        
    def get_capacity(self) -> int:
        """
        Get the current capacity of the internal array.
        
        This method is useful for performance analysis and understanding
        the memory usage characteristics of the queue.
        
        Returns:
            The current capacity of the internal array.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> queue = Queue[int]()
            >>> queue.get_capacity()
            8
            >>> for i in range(10):
            ...     queue.enqueue(i)
            >>> queue.get_capacity()  # Automatically resized
            16
        """
        return self._capacity
        
    def get_load_factor(self) -> float:
        """
        Get the current load factor (utilization) of the queue.
        
        The load factor is the ratio of current size to capacity.
        This is useful for performance analysis and memory optimization.
        
        Returns:
            The load factor as a float between 0.0 and 1.0.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> queue = Queue[str]()
            >>> queue.enqueue("a")
            >>> queue.enqueue("b")
            >>> round(queue.get_load_factor(), 2)
            0.25
        """
        return self._size / self._capacity if self._capacity > 0 else 0.0
        
    def to_list(self) -> list[T]:
        """
        Convert the queue to a list in FIFO order.
        
        This method creates a new list containing all queue elements
        without modifying the original queue.
        
        Returns:
            A list containing all queue elements in FIFO order.
            
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Example:
            >>> queue = Queue[int]()
            >>> queue.enqueue(1)
            >>> queue.enqueue(2)
            >>> queue.enqueue(3)
            >>> queue.to_list()
            [1, 2, 3]
        """
        return list(self)

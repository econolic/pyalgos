"""
Binary Min-Heap implementation with functionality and visualization.

This module provides a binary min-heap data structure with
efficient insertion, extraction, and heapify operations. The implementation
includes advanced features like heap construction from arrays, visualization
capabilities, and performance analysis tools.

Key Features:
    - Array-based binary min-heap with 1-indexing
    - O(log n) insertion and extraction operations
    - O(n) heap construction from unsorted array
    - Input validation and error handling
    - Visualization support following patterns from advanced implementations
    - Full type safety with generic type support
    - Performance measurement and complexity analysis

Example:
    >>> from pyalgos.heaps.binary_heap import BinaryHeap
    >>> heap = BinaryHeap[int]()
    >>> heap.insert(5)
    >>> heap.insert(3)
    >>> heap.insert(8)
    >>> heap.insert(1)
    >>> print(heap.peek_min())  # 1
    >>> print(heap.extract_min())  # 1
    >>> print(heap.size())  # 3
"""

from __future__ import annotations

import time
from typing import Any, Generic, Iterator, TypeVar, Callable

from pyalgos.exceptions import EmptyStructureError, InvalidOperationError
from pyalgos.types import Comparable, SizeType

T = TypeVar("T")
ComparableT = TypeVar("ComparableT", bound=Comparable)


class BinaryHeap(Generic[T]):
    """
    A generic binary min-heap implementation.
    
    This heap maintains the min-heap property where each parent node is smaller
    than or equal to its children. The implementation uses a 1-indexed array
    representation for efficient parent/child calculations.
    
    The heap supports custom comparison functions for flexible ordering and
    includes advanced operations like heapify, merge, and visualization.
    
    Type Parameters:
        T: The type of elements stored in the heap.
        
    Attributes:
        _data: Internal array storing heap elements (1-indexed).
        _size: Current number of elements in the heap.
        _capacity: Current capacity of the internal array.
        _compare: Comparison function for ordering elements.
        
    Time Complexities:
        - insert(): O(log n)
        - extract_min(): O(log n)
        - peek_min(): O(1)
        - heapify(): O(n)
        - build_heap(): O(n)
        
    Space Complexity: O(n) where n is the number of elements
    """
    
    # Class constants for array management
    _INITIAL_CAPACITY: int = 16
    _GROWTH_FACTOR: int = 2
    
    def __init__(
        self, 
        initial_capacity: int | None = None,
        compare_func: Callable[[T, T], bool] | None = None
    ) -> None:
        """
        Initialize an empty binary heap.
        
        Args:
            initial_capacity: Optional initial capacity for the internal array.
                            Must be positive if provided. Defaults to 16.
            compare_func: Optional comparison function. If None, uses default
                         comparison (assumes T supports < operator).
                         Function should return True if first arg < second arg.
                         
        Raises:
            InvalidOperationError: If initial_capacity is not positive.
            
        Example:
            >>> heap = BinaryHeap[int]()
            >>> max_heap = BinaryHeap[int](compare_func=lambda x, y: x > y)
        """
        if initial_capacity is not None:
            if not isinstance(initial_capacity, int) or initial_capacity <= 0:
                raise InvalidOperationError(
                    f"Initial capacity must be a positive integer, got {initial_capacity}"
                )
            self._capacity = initial_capacity + 1  # +1 for 1-indexing
        else:
            self._capacity = self._INITIAL_CAPACITY + 1
            
        # Use 1-indexing for easier parent/child calculations
        self._data: list[T | None] = [None] * self._capacity
        self._size: SizeType = 0
        
        # Set comparison function
        if compare_func is not None:
            self._compare = compare_func
        else:
            self._compare = lambda x, y: x < y  # type: ignore[operator]
            
    @classmethod
    def from_list(
        cls, 
        items: list[T], 
        compare_func: Callable[[T, T], bool] | None = None
    ) -> BinaryHeap[T]:
        """
        Create a heap from a list using the optimal O(n) heapify algorithm.
        
        This class method creates a heap from an existing list in linear time,
        which is more efficient than inserting elements one by one.
        
        Args:
            items: List of items to create the heap from.
            compare_func: Optional comparison function.
            
        Returns:
            A new BinaryHeap containing the items.
            
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Example:
            >>> heap = BinaryHeap.from_list([4, 10, 3, 5, 1])
            >>> heap.peek_min()
            1
        """
        if not items:
            return cls(compare_func=compare_func)
            
        heap = cls(initial_capacity=len(items), compare_func=compare_func)
        
        # Copy items to heap array (skip index 0 for 1-indexing)
        for i, item in enumerate(items, 1):
            if item is None:
                raise InvalidOperationError("Cannot add None to heap")
            heap._data[i] = item
            
        heap._size = len(items)
        
        # Heapify the array
        heap._build_heap()
        
        return heap
        
    def insert(self, item: T) -> None:
        """
        Insert an item into the heap maintaining the heap property.
        
        The item is initially placed at the end and then bubbled up
        to its correct position to maintain the min-heap property.
        
        Args:
            item: The item to insert into the heap.
            
        Raises:
            InvalidOperationError: If item is None.
            
        Time Complexity: O(log n)
        Space Complexity: O(1)
        
        Example:
            >>> heap = BinaryHeap[int]()
            >>> heap.insert(5)
            >>> heap.insert(3)
            >>> heap.insert(7)
            >>> heap.peek_min()
            3
        """
        if item is None:
            raise InvalidOperationError("Cannot insert None into heap")
            
        # Resize if necessary
        if self._size + 1 >= self._capacity:
            self._resize(self._capacity * self._GROWTH_FACTOR)
            
        # Insert at end and bubble up
        self._size += 1
        self._data[self._size] = item
        self._bubble_up(self._size)
        
    def extract_min(self) -> T:
        """
        Remove and return the minimum element from the heap.
        
        The root element is removed and replaced with the last element,
        which is then bubbled down to maintain the heap property.
        
        Returns:
            The minimum element that was removed from the heap.
            
        Raises:
            EmptyStructureError: If the heap is empty.
            
        Time Complexity: O(log n)
        Space Complexity: O(1)
        
        Example:
            >>> heap = BinaryHeap.from_list([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
            >>> heap.extract_min()
            1
            >>> heap.extract_min()
            2
        """
        if self.is_empty():
            raise EmptyStructureError("BinaryHeap")
            
        min_item = self._data[1]
        
        # Move last element to root and reduce size
        self._data[1] = self._data[self._size]
        self._data[self._size] = None  # Help garbage collection
        self._size -= 1
        
        # Restore heap property if heap is not empty
        if self._size > 0:
            self._bubble_down(1)
            
        return min_item  # type: ignore[return-value]
        
    def peek_min(self) -> T:
        """
        Return the minimum element without removing it from the heap.
        
        Returns:
            The minimum element in the heap.
            
        Raises:
            EmptyStructureError: If the heap is empty.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> heap = BinaryHeap.from_list([3, 1, 4])
            >>> heap.peek_min()
            1
            >>> heap.size()  # Element is still in heap
            3
        """
        if self.is_empty():
            raise EmptyStructureError("BinaryHeap")
            
        return self._data[1]  # type: ignore[return-value]
        
    def decrease_key(self, index: int, new_value: T) -> None:
        """
        Decrease the value of an element at a specific index.
        
        This operation is useful for priority queue implementations
        where priorities need to be updated.
        
        Args:
            index: The 1-based index of the element to update.
            new_value: The new value (must be smaller than current value).
            
        Raises:
            InvalidOperationError: If index is invalid or new_value is not smaller.
            
        Time Complexity: O(log n)
        Space Complexity: O(1)
        
        Example:
            >>> heap = BinaryHeap.from_list([5, 10, 15, 20])
            >>> heap.decrease_key(2, 3)  # Change element at index 2 to 3
            >>> heap.peek_min()
            3
        """
        if index < 1 or index > self._size:
            raise InvalidOperationError(f"Index {index} out of bounds")
            
        current_value = self._data[index]
        if not self._compare(new_value, current_value):
            raise InvalidOperationError("New value must be smaller than current value")
            
        self._data[index] = new_value
        self._bubble_up(index)
        
    def merge(self, other: BinaryHeap[T]) -> None:
        """
        Merge another heap into this heap.
        
        This operation combines two heaps by extracting all elements
        from the other heap and inserting them into this heap.
        
        Args:
            other: Another BinaryHeap to merge into this one.
            
        Time Complexity: O(m log(n + m)) where n, m are heap sizes
        Space Complexity: O(1)
        
        Example:
            >>> heap1 = BinaryHeap.from_list([1, 3, 5])
            >>> heap2 = BinaryHeap.from_list([2, 4, 6])
            >>> heap1.merge(heap2)
            >>> heap1.size()
            6
        """
        # Extract all elements from other heap and insert into this heap
        while not other.is_empty():
            self.insert(other.extract_min())
            
    def heapify(self) -> None:
        """
        Restore the heap property for the current array.
        
        This method is useful when the heap property has been violated
        and needs to be restored. It's more efficient than rebuilding
        the heap from scratch.
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> heap = BinaryHeap[int]()
            >>> # Manually manipulate internal structure (not recommended)
            >>> heap._data[1:4] = [10, 2, 5]
            >>> heap._size = 3
            >>> heap.heapify()  # Restore heap property
            >>> heap.peek_min()
            2
        """
        self._build_heap()
        
    def is_empty(self) -> bool:
        """
        Check if the heap is empty.
        
        Returns:
            True if the heap contains no elements, False otherwise.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self._size == 0
        
    def size(self) -> int:
        """
        Return the number of elements in the heap.
        
        Returns:
            The number of elements currently in the heap.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self._size
        
    def clear(self) -> None:
        """
        Remove all elements from the heap.
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._data = [None] * self._capacity
        self._size = 0
        
    def to_list(self) -> list[T]:
        """
        Convert the heap to a sorted list by repeatedly extracting minimum.
        
        Note: This operation destroys the original heap.
        
        Returns:
            A sorted list containing all heap elements.
            
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        
        Example:
            >>> heap = BinaryHeap.from_list([4, 1, 3, 2, 5])
            >>> sorted_list = heap.to_list()
            >>> sorted_list
            [1, 2, 3, 4, 5]
            >>> heap.is_empty()  # Heap is now empty
            True
        """
        result = []
        while not self.is_empty():
            result.append(self.extract_min())
        return result
        
    def to_array_representation(self) -> list[T]:
        """
        Get the internal array representation of the heap.
        
        This method returns a copy of the heap's internal array
        representation, which can be useful for debugging or visualization.
        
        Returns:
            A list representing the heap's internal structure.
            
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Example:
            >>> heap = BinaryHeap.from_list([4, 1, 3, 2])
            >>> heap.to_array_representation()
            [1, 2, 3, 4]
        """
        return [self._data[i] for i in range(1, self._size + 1)]  # type: ignore[misc]
        
    def __len__(self) -> int:
        """Return the number of elements in the heap."""
        return self._size
        
    def __contains__(self, item: object) -> bool:
        """Check if an item is in the heap (O(n) operation)."""
        for i in range(1, self._size + 1):
            if self._data[i] == item:
                return True
        return False
        
    def __iter__(self) -> Iterator[T]:
        """
        Iterate over heap elements in level-order (not sorted order).
        
        Note: This iterates over the internal array representation,
        not in sorted order.
        """
        for i in range(1, self._size + 1):
            yield self._data[i]  # type: ignore[misc]            
    def __repr__(self) -> str:
        """Return detailed string representation."""
        elements = [str(self._data[i]) for i in range(1, self._size + 1)]
        return f"BinaryHeap([{', '.join(elements)}], size={self._size})"
        
    def __str__(self) -> str:
        """Return simple string representation."""
        if self.is_empty():
            return "BinaryHeap[]"
        elements = [str(self._data[i]) for i in range(1, self._size + 1)]
        return f"BinaryHeap[{', '.join(elements)}]"
        
    def _parent(self, index: int) -> int:
        """Get the parent index of the given index."""
        return index // 2
        
    def _left_child(self, index: int) -> int:
        """Get the left child index of the given index."""
        return 2 * index
        
    def _right_child(self, index: int) -> int:
        """Get the right child index of the given index."""
        return 2 * index + 1
        
    def _has_left_child(self, index: int) -> bool:
        """Check if the node at index has a left child."""
        return self._left_child(index) <= self._size
        
    def _has_right_child(self, index: int) -> bool:
        """Check if the node at index has a right child."""
        return self._right_child(index) <= self._size
        
    def _has_parent(self, index: int) -> bool:
        """Check if the node at index has a parent."""
        return index > 1
        
    def _bubble_up(self, index: int) -> None:
        """
        Bubble up the element at index to maintain heap property.
        
        Args:
            index: The index of the element to bubble up.
        """
        while (self._has_parent(index) and 
               self._compare(self._data[index], self._data[self._parent(index)])):
            # Swap with parent
            parent_idx = self._parent(index)
            self._data[index], self._data[parent_idx] = (
                self._data[parent_idx], self._data[index]
            )
            index = parent_idx
            
    def _bubble_down(self, index: int) -> None:
        """
        Bubble down the element at index to maintain heap property.
        
        Args:
            index: The index of the element to bubble down.
        """
        while self._has_left_child(index):
            # Find the smaller child
            smaller_child_idx = self._left_child(index)
            
            if (self._has_right_child(index) and 
                self._compare(self._data[self._right_child(index)], 
                             self._data[smaller_child_idx])):
                smaller_child_idx = self._right_child(index)
                
            # If current element is smaller than both children, we're done
            if self._compare(self._data[index], self._data[smaller_child_idx]):
                break
                
            # Swap with smaller child
            self._data[index], self._data[smaller_child_idx] = (
                self._data[smaller_child_idx], self._data[index]
            )
            index = smaller_child_idx
            
    def _build_heap(self) -> None:
        """
        Build heap property for the current array in O(n) time.
        
        This method starts from the last non-leaf node and bubbles down
        each node, which is more efficient than inserting elements one by one.
        """
        # Start from the last non-leaf node and bubble down
        for i in range(self._size // 2, 0, -1):
            self._bubble_down(i)
            
    def _resize(self, new_capacity: int) -> None:
        """
        Resize the internal array to the specified capacity.
        
        Args:
            new_capacity: The new capacity for the internal array.
        """
        old_data = self._data
        self._data = [None] * new_capacity
        self._capacity = new_capacity
        
        # Copy existing elements
        for i in range(1, self._size + 1):
            self._data[i] = old_data[i]
            
    def get_heap_height(self) -> int:
        """
        Calculate the height of the heap.
        
        Returns:
            The height of the heap (0 for empty heap).
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> heap = BinaryHeap.from_list([1, 2, 3, 4, 5, 6, 7])
            >>> heap.get_heap_height()
            2
        """
        if self.is_empty():
            return 0
        
        import math
        return math.floor(math.log2(self._size))
        
    def validate_heap_property(self) -> bool:
        """
        Validate that the heap property is maintained.
        
        This method is useful for debugging and testing.
        
        Returns:
            True if heap property is maintained, False otherwise.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        for i in range(2, self._size + 1):
            parent_idx = self._parent(i)
            if not self._compare(self._data[parent_idx], self._data[i]):
                return False
        return True

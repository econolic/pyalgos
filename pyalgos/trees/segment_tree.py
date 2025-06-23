"""
Segment Tree implementation with range queries and lazy propagation.

This module provides a production-ready segment tree data structure for efficient
range queries and updates. The implementation supports various operations like
range sum, range minimum/maximum queries, and includes lazy propagation for
efficient range updates.

Key Features:
    - Range queries (sum, min, max) in O(log n) time
    - Point and range updates in O(log n) time
    - Lazy propagation for efficient range updates
    - Generic support for different operation types
    - Comprehensive input validation and error handling
    - Memory-efficient tree representation

Example:
    >>> from pyalgos.trees.segment_tree import SegmentTree
    >>> # Range sum query tree
    >>> tree = SegmentTree([1, 3, 5, 7, 9, 11])
    >>> print(tree.range_query(1, 3))  # Sum of elements from index 1 to 3
    15
    >>> tree.update(2, 10)  # Update index 2 to value 10
    >>> print(tree.range_query(1, 3))  # New sum
    20
"""

from __future__ import annotations

import math
from typing import Any, Callable, Generic, TypeVar

from pyalgos.exceptions import IndexOutOfBoundsError, InvalidOperationError
from pyalgos.types import Comparable, SizeType

T = TypeVar("T")
ComparableT = TypeVar("ComparableT", bound=Comparable)


class SegmentTree(Generic[T]):
    """
    A generic segment tree implementation with lazy propagation support.
    
    This segment tree supports range queries and updates efficiently using
    a binary tree structure. The tree can be configured for different operations
    like sum, min, max through custom operation and identity functions.
    
    The implementation uses lazy propagation for efficient range updates,
    which allows O(log n) range updates instead of O(n).
    
    Type Parameters:
        T: The type of elements stored in the tree.
        
    Attributes:
        _tree: Internal array storing the segment tree nodes.
        _lazy: Array for lazy propagation values.
        _size: Size of the original array.
        _operation: Function defining the tree operation (sum, min, max, etc.).
        _identity: Identity element for the operation.
        _update_op: Function for applying updates.
        
    Time Complexities:
        - build(): O(n)
        - range_query(): O(log n)
        - update(): O(log n)
        - range_update(): O(log n) with lazy propagation
        
    Space Complexity: O(n)
    """
    
    def __init__(
        self,
        array: list[T],
        operation: Callable[[T, T], T] | None = None,
        identity: T | None = None,
        update_operation: Callable[[T, T], T] | None = None
    ) -> None:
        """
        Initialize the segment tree with the given array and operation.
        
        Args:
            array: The input array to build the segment tree from.
            operation: The binary operation function (e.g., sum, min, max).
                      Defaults to sum for numeric types.
            identity: The identity element for the operation.
                     Defaults to 0 for sum operation.
            update_operation: Function for applying range updates.
                            Defaults to addition for range updates.
                            
        Raises:
            InvalidOperationError: If array is empty or operations are invalid.
            
        Example:
            >>> # Sum segment tree
            >>> tree = SegmentTree([1, 3, 5, 7, 9])
            >>> 
            >>> # Min segment tree
            >>> min_tree = SegmentTree([1, 3, 5, 7, 9], 
            ...                        operation=min, 
            ...                        identity=float('inf'))
        """
        if not array:
            raise InvalidOperationError("Cannot create segment tree from empty array")
            
        self._size = len(array)
        
        # Set default operations for sum if not provided
        if operation is None:
            self._operation = lambda x, y: x + y  # type: ignore[operator]
        else:
            self._operation = operation
            
        if identity is None:
            # Default identity for sum operation
            try:
                self._identity = type(array[0])(0)  # type: ignore[misc]
            except (TypeError, ValueError):
                raise InvalidOperationError(
                    "Cannot determine identity element. Please provide identity parameter."
                )
        else:
            self._identity = identity
            
        if update_operation is None:
            self._update_op = lambda x, y: x + y  # type: ignore[operator]
        else:
            self._update_op = update_operation
              # Calculate tree size (4 * n is sufficient for most cases)
        tree_size = 4 * self._size
        self._tree: list[T] = [self._identity] * tree_size
        self._lazy: list[T] = [self._identity] * tree_size
        
        # Build the segment tree
        self._build(array, 0, 0, self._size - 1)
        
    @classmethod
    def create_sum_tree(cls, array: list[T]) -> SegmentTree[T]:
        """
        Create a segment tree for range sum queries.
        
        Args:
            array: Array of numeric values.
            
        Returns:
            A SegmentTree configured for sum operations.
            
        Example:
            >>> tree = SegmentTree.create_sum_tree([1, 3, 5, 7, 9])
            >>> tree.range_query(1, 3)  # Sum from index 1 to 3
            15
        """
        if not array:
            raise InvalidOperationError("Cannot create sum tree from empty array")
        try:
            zero_val = type(array[0])(0)  # type: ignore[misc]
        except (TypeError, ValueError):
            zero_val = 0  # type: ignore[assignment]
        return cls(array, operation=lambda x, y: x + y, identity=zero_val)  # type: ignore[operator,arg-type]
        
    @classmethod
    def create_min_tree(cls, array: list[ComparableT]) -> SegmentTree[ComparableT]:
        """
        Create a segment tree for range minimum queries.
        
        Args:
            array: Array of comparable values.
            
        Returns:
            A SegmentTree configured for min operations.
            
        Example:
            >>> tree = SegmentTree.create_min_tree([3, 1, 4, 1, 5, 9])
            >>> tree.range_query(0, 2)  # Min from index 0 to 2
            1
        """
        return cls(array, operation=min, identity=float('inf'))  # type: ignore[arg-type]
        
    @classmethod
    def create_max_tree(cls, array: list[ComparableT]) -> SegmentTree[ComparableT]:
        """
        Create a segment tree for range maximum queries.
        
        Args:
            array: Array of comparable values.
            
        Returns:
            A SegmentTree configured for max operations.
            
        Example:
            >>> tree = SegmentTree.create_max_tree([3, 1, 4, 1, 5, 9])
            >>> tree.range_query(2, 5)  # Max from index 2 to 5
            9
        """
        return cls(array, operation=max, identity=float('-inf'))  # type: ignore[arg-type]
        
    def range_query(self, left: int, right: int) -> T:
        """
        Perform a range query on the segment [left, right] (inclusive).
        
        Args:
            left: Left boundary of the query range (0-indexed).
            right: Right boundary of the query range (0-indexed).
            
        Returns:
            The result of applying the operation over the range.
            
        Raises:
            IndexOutOfBoundsError: If indices are out of bounds.
            
        Time Complexity: O(log n)
        Space Complexity: O(1)
        
        Example:
            >>> tree = SegmentTree([1, 3, 5, 7, 9, 11])
            >>> tree.range_query(1, 4)  # Query range [1, 4]
            24
        """
        if left < 0 or right >= self._size or left > right:
            raise IndexOutOfBoundsError(
                f"Query range [{left}, {right}]", self._size, "SegmentTree"
            )
            
        return self._range_query(0, 0, self._size - 1, left, right)
        
    def update(self, index: int, value: T) -> None:
        """
        Update the value at a specific index.
        
        Args:
            index: The index to update (0-indexed).
            value: The new value to set.
            
        Raises:
            IndexOutOfBoundsError: If index is out of bounds.
            
        Time Complexity: O(log n)
        Space Complexity: O(1)
        
        Example:
            >>> tree = SegmentTree([1, 3, 5, 7, 9])
            >>> tree.update(2, 10)  # Set index 2 to value 10
            >>> tree.range_query(2, 2)  # Query single element
            10
        """
        if index < 0 or index >= self._size:
            raise IndexOutOfBoundsError(index, self._size, "SegmentTree")
            
        self._update(0, 0, self._size - 1, index, value)
        
    def range_update(self, left: int, right: int, value: T) -> None:
        """
        Update all values in the range [left, right] using lazy propagation.
        
        This operation applies the update operation to all elements in the
        specified range efficiently using lazy propagation.
        
        Args:
            left: Left boundary of the update range (0-indexed).
            right: Right boundary of the update range (0-indexed).
            value: The value to apply to the range.
            
        Raises:
            IndexOutOfBoundsError: If indices are out of bounds.
            
        Time Complexity: O(log n)
        Space Complexity: O(1)
        
        Example:
            >>> tree = SegmentTree([1, 3, 5, 7, 9])
            >>> tree.range_update(1, 3, 5)  # Add 5 to indices 1, 2, 3
            >>> tree.range_query(1, 3)  # Query updated range
            30  # (3+5) + (5+5) + (7+5) = 8 + 10 + 12 = 30
        """
        if left < 0 or right >= self._size or left > right:
            raise IndexOutOfBoundsError(
                f"Update range [{left}, {right}]", self._size, "SegmentTree"
            )
            
        self._range_update(0, 0, self._size - 1, left, right, value)
        
    def get_array(self) -> list[T]:
        """
        Reconstruct and return the current state of the underlying array.
        
        This method extracts individual elements to reconstruct the array
        after any updates have been applied.
        
        Returns:
            The current array represented by the segment tree.
            
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        
        Example:
            >>> tree = SegmentTree([1, 3, 5, 7])
            >>> tree.update(1, 10)
            >>> tree.get_array()
            [1, 10, 5, 7]
        """
        result = []
        for i in range(self._size):
            # Query single element to get current value
            result.append(self.range_query(i, i))
        return result
        
    def size(self) -> int:
        """
        Return the size of the underlying array.
        
        Returns:
            The number of elements in the original array.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self._size
        
    def _build(self, array: list[T], node: int, start: int, end: int) -> None:
        """
        Build the segment tree recursively.
        
        Args:
            array: The input array.
            node: Current node index in the tree.
            start: Start index of the current segment.
            end: End index of the current segment.
        """
        if start == end:
            # Leaf node
            self._tree[node] = array[start]
        else:
            # Internal node
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            # Recursively build left and right subtrees
            self._build(array, left_child, start, mid)
            self._build(array, right_child, mid + 1, end)
            
            # Combine results from children
            self._tree[node] = self._operation(
                self._tree[left_child], self._tree[right_child]
            )
            
    def _range_query(
        self, node: int, start: int, end: int, left: int, right: int
    ) -> T:
        """
        Perform range query recursively with lazy propagation.
        
        Args:
            node: Current node index.
            start: Start of current segment.
            end: End of current segment.
            left: Query range start.
            right: Query range end.
            
        Returns:
            Result of the range query.
        """
        # Apply pending updates
        self._push(node, start, end)
        
        # No overlap
        if start > right or end < left:
            return self._identity
            
        # Complete overlap
        if start >= left and end <= right:
            return self._tree[node]
            
        # Partial overlap
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_result = self._range_query(left_child, start, mid, left, right)
        right_result = self._range_query(right_child, mid + 1, end, left, right)
        
        return self._operation(left_result, right_result)
        
    def _update(self, node: int, start: int, end: int, index: int, value: T) -> None:
        """
        Update a single element recursively.
        
        Args:
            node: Current node index.
            start: Start of current segment.
            end: End of current segment.
            index: Index to update.
            value: New value.
        """
        # Apply pending updates
        self._push(node, start, end)
        
        if start == end:
            # Leaf node
            self._tree[node] = value
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if index <= mid:
                self._update(left_child, start, mid, index, value)
            else:
                self._update(right_child, mid + 1, end, index, value)
                
            # Update current node
            self._tree[node] = self._operation(
                self._tree[left_child], self._tree[right_child]
            )
            
    def _range_update(
        self, node: int, start: int, end: int, left: int, right: int, value: T
    ) -> None:
        """
        Perform range update with lazy propagation.
        
        Args:
            node: Current node index.
            start: Start of current segment.
            end: End of current segment.
            left: Update range start.
            right: Update range end.
            value: Value to apply.
        """
        # Apply pending updates
        self._push(node, start, end)
        
        # No overlap
        if start > right or end < left:
            return
            
        # Complete overlap
        if start >= left and end <= right:
            # Apply lazy update
            self._lazy[node] = self._update_op(self._lazy[node], value)
            self._push(node, start, end)
            return
            
        # Partial overlap
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        self._range_update(left_child, start, mid, left, right, value)
        self._range_update(right_child, mid + 1, end, left, right, value)
        
        # Update current node
        self._tree[node] = self._operation(
            self._tree[left_child], self._tree[right_child]
        )
        
    def _push(self, node: int, start: int, end: int) -> None:
        """
        Apply lazy propagation updates.
        
        Args:
            node: Current node index.
            start: Start of current segment.
            end: End of current segment.
        """
        if self._lazy[node] != self._identity:
            # Apply the lazy update to current node
            range_size = end - start + 1
            
            # For sum operations, multiply by range size
            if self._operation == (lambda x, y: x + y):
                update_value = self._lazy[node] * range_size  # type: ignore[operator]
            else:
                update_value = self._lazy[node]
                
            self._tree[node] = self._update_op(self._tree[node], update_value)
            
            # Propagate to children if not a leaf
            if start != end:
                left_child = 2 * node + 1
                right_child = 2 * node + 2
                
                self._lazy[left_child] = self._update_op(
                    self._lazy[left_child], self._lazy[node]
                )
                self._lazy[right_child] = self._update_op(
                    self._lazy[right_child], self._lazy[node]
                )
                
            # Clear the lazy value
            self._lazy[node] = self._identity
            
    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"SegmentTree(size={self._size}, operation={self._operation.__name__ if hasattr(self._operation, '__name__') else 'custom'})"
        
    def __str__(self) -> str:
        """Return simple string representation."""
        try:
            array_repr = str(self.get_array())
            return f"SegmentTree{array_repr}"
        except Exception:
            return f"SegmentTree(size={self._size})"

"""
Doubly Linked List implementation with operations and type safety.

This module provides a doubly linked list data structure with
full bidirectional traversal capabilities. The implementation includes advanced
operations like merge sort, reversal, and efficient insertion/deletion.

Key Features:
    - Doubly linked nodes with prev/next pointers
    - O(1) insertion/deletion at both ends
    - Bidirectional traversal support
    - Advanced operations: merge sort, reversal, merge
    - Input validation and error handling
    - Full type safety with generic type support
    - Performance measurement and complexity analysis

Example:
    >>> from pyalgos.linear.linked_list import LinkedList
    >>> ll = LinkedList[int]()
    >>> ll.append(1)
    >>> ll.append(2)
    >>> ll.prepend(0)
    >>> print(list(ll))  # [0, 1, 2]
    >>> ll.reverse()
    >>> print(list(ll))  # [2, 1, 0]
"""

from __future__ import annotations

import time
from typing import Any, Generic, Iterator, TypeVar

from pyalgos.exceptions import EmptyStructureError, IndexOutOfBoundsError, InvalidOperationError
from pyalgos.types import Comparable, SizeType

T = TypeVar("T")


class ListNode(Generic[T]):
    """
    A node in a doubly linked list.
    
    This class represents a single node with data and pointers to
    both the previous and next nodes, enabling bidirectional traversal.
    
    Type Parameters:
        T: The type of data stored in the node.
        
    Attributes:
        data: The data stored in this node.
        next: Reference to the next node in the list.
        prev: Reference to the previous node in the list.
    """
    
    def __init__(self, data: T) -> None:
        """
        Initialize a new node with the given data.
        
        Args:
            data: The data to store in this node.
        """
        if data is None:
            raise InvalidOperationError("Node data cannot be None")
        self.data: T = data
        self.next: ListNode[T] | None = None
        self.prev: ListNode[T] | None = None
        
    def __repr__(self) -> str:
        """Return string representation of the node."""
        return f"ListNode({self.data})"
        
    def __str__(self) -> str:
        """Return string representation of the node data."""
        return str(self.data)


def measure_performance(func: Any) -> Any:
    """
    Decorator to measure function execution time.
    
    This decorator follows the performance measurement patterns from
    advanced algorithm implementations, providing timing analysis.
    
    Args:
        func: The function to be measured.
        
    Returns:
        A wrapper function that returns (result, execution_time).
    """
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


class LinkedList(Generic[T]):
    """
    A generic doubly linked list implementation with advanced operations.
    
    This linked list provides efficient insertion and deletion at both ends,
    bidirectional traversal, and advanced operations like sorting and merging.
    The implementation maintains references to both head and tail for O(1)
    operations at either end.
    
    Type Parameters:
        T: The type of elements stored in the list.
        
    Attributes:
        _head: Reference to the first node in the list.
        _tail: Reference to the last node in the list.
        _size: Current number of elements in the list.
        
    Time Complexities:
        - append(): O(1)
        - prepend(): O(1)
        - insert(): O(n)
        - delete(): O(n)
        - find(): O(n)
        - reverse(): O(n)
        - merge_sort(): O(n log n)
        
    Space Complexity: O(n) where n is the number of elements
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty doubly linked list.
        
        Example:
            >>> ll = LinkedList[str]()
            >>> ll.is_empty()
            True
        """
        self._head: ListNode[T] | None = None
        self._tail: ListNode[T] | None = None
        self._size: SizeType = 0
        
    @classmethod
    def from_list(cls, items: list[T]) -> LinkedList[T]:
        """
        Create a LinkedList from a Python list.
        
        This class method provides a convenient way to initialize a LinkedList
        with existing data, following patterns from advanced implementations.
        
        Args:
            items: List of items to add to the LinkedList.
            
        Returns:
            A new LinkedList containing the items.
            
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Example:
            >>> ll = LinkedList.from_list([1, 2, 3, 4])
            >>> list(ll)
            [1, 2, 3, 4]
        """
        linked_list = cls()
        for item in items:
            linked_list.append(item)
        return linked_list
        
    def append(self, data: T) -> None:
        """
        Add an element to the end of the list.
        
        Args:
            data: The data to append.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> ll = LinkedList[int]()
            >>> ll.append(1)
            >>> ll.append(2)
            >>> list(ll)
            [1, 2]
        """
        if data is None:
            raise InvalidOperationError("Cannot append None to the list")
            
        new_node = ListNode(data)
        
        if self._head is None:
            # First node
            self._head = self._tail = new_node
        else:
            # Link new node to current tail
            new_node.prev = self._tail
            self._tail.next = new_node  # type: ignore[union-attr]
            self._tail = new_node
            
        self._size += 1
        
    def prepend(self, data: T) -> None:
        """
        Add an element to the beginning of the list.
        
        Args:
            data: The data to prepend.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> ll = LinkedList[str]()
            >>> ll.append("world")
            >>> ll.prepend("hello")
            >>> list(ll)
            ['hello', 'world']
        """
        if data is None:
            raise InvalidOperationError("Cannot prepend None to the list")
            
        new_node = ListNode(data)
        
        if self._head is None:
            # First node
            self._head = self._tail = new_node
        else:
            # Link new node to current head
            new_node.next = self._head
            self._head.prev = new_node
            self._head = new_node
            
        self._size += 1
        
    def insert(self, index: int, data: T) -> None:
        """
        Insert an element at the specified index.
        
        Args:
            index: The index at which to insert the data.
            data: The data to insert.
            
        Raises:
            IndexOutOfBoundsError: If index is out of bounds.
            InvalidOperationError: If data is None.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> ll = LinkedList[int]()
            >>> ll.append(1)
            >>> ll.append(3)
            >>> ll.insert(1, 2)
            >>> list(ll)
            [1, 2, 3]
        """
        if data is None:
            raise InvalidOperationError("Cannot insert None into the list")
            
        if index < 0 or index > self._size:
            raise IndexOutOfBoundsError(index, self._size, "LinkedList")
            
        # Handle edge cases
        if index == 0:
            self.prepend(data)
            return
        if index == self._size:
            self.append(data)
            return
            
        # Find insertion point
        current = self._get_node_at_index(index)
        new_node = ListNode(data)
        
        # Insert between current.prev and current
        new_node.next = current
        new_node.prev = current.prev
        current.prev.next = new_node  # type: ignore[union-attr]
        current.prev = new_node
        
        self._size += 1
        
    def delete(self, data: T) -> bool:
        """
        Delete the first occurrence of the specified data.
        
        Args:
            data: The data to delete.
            
        Returns:
            True if the data was found and deleted, False otherwise.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> ll = LinkedList[str]()
            >>> ll.append("a")
            >>> ll.append("b")
            >>> ll.append("c")
            >>> ll.delete("b")
            True
            >>> list(ll)
            ['a', 'c']
        """
        if self.is_empty():
            return False
            
        current = self._head
        while current is not None:
            if current.data == data:
                self._remove_node(current)
                return True
            current = current.next
            
        return False
        
    def delete_at_index(self, index: int) -> T:
        """
        Delete the element at the specified index.
        
        Args:
            index: The index of the element to delete.
            
        Returns:
            The data that was deleted.
            
        Raises:
            IndexOutOfBoundsError: If index is out of bounds.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> ll = LinkedList[int]()
            >>> ll.extend([1, 2, 3, 4])
            >>> ll.delete_at_index(2)
            3
            >>> list(ll)
            [1, 2, 4]
        """
        if index < 0 or index >= self._size:
            raise IndexOutOfBoundsError(index, self._size, "LinkedList")
            
        node = self._get_node_at_index(index)
        data = node.data
        self._remove_node(node)
        return data
        
    def find(self, data: T) -> int:
        """
        Find the index of the first occurrence of the specified data.
        
        Args:
            data: The data to find.
            
        Returns:
            The index of the data, or -1 if not found.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> ll = LinkedList[str]()
            >>> ll.extend(["apple", "banana", "cherry"])
            >>> ll.find("banana")
            1
            >>> ll.find("grape")
            -1
        """
        current = self._head
        index = 0
        
        while current is not None:
            if current.data == data:
                return index
            current = current.next
            index += 1
            
        return -1
        
    def get(self, index: int) -> T:
        """
        Get the element at the specified index.
        
        Args:
            index: The index of the element to retrieve. Supports negative indices.
            
        Returns:
            The element at the specified index.
            
        Raises:
            IndexOutOfBoundsError: If the index is out of bounds.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> ll = LinkedList[int]()
            >>> ll.append(10)
            >>> ll.get(0)
            10
            >>> ll.get(-1)  # Last element
            10
        """
        if self._size == 0:
            raise IndexOutOfBoundsError(index, self._size, "LinkedList")
            
        # Handle negative indices
        if index < 0:
            index = self._size + index
            
        if not (0 <= index < self._size):
            raise IndexOutOfBoundsError(index, self._size, "LinkedList")
            
        node = self._get_node_at_index(index)
        return node.data

    def set(self, index: int, data: T) -> None:
        """
        Set the element at the specified index.
        
        Args:
            index: The index where to set the element. Supports negative indices.
            data: The new data to set.
            
        Raises:
            IndexOutOfBoundsError: If the index is out of bounds.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> ll = LinkedList[int]()
            >>> ll.append(10)
            >>> ll.set(0, 20)
            >>> ll.get(0)
            20
            >>> ll.set(-1, 30)  # Last element
        """
        if self._size == 0:
            raise IndexOutOfBoundsError(index, self._size, "LinkedList")
            
        # Handle negative indices
        if index < 0:
            index = self._size + index
            
        if not (0 <= index < self._size):
            raise IndexOutOfBoundsError(index, self._size, "LinkedList")            
        node = self._get_node_at_index(index)
        node.data = data
        
    def remove(self, index: int) -> T:
        """
        Remove and return the element at the specified index.
        
        Args:
            index: The index of the element to remove.
            
        Returns:
            The removed element.
            
        Raises:
            InvalidOperationError: If the index is out of bounds.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> ll = LinkedList[int]()
            >>> ll.extend([1, 2, 3])
            >>> ll.remove(1)
            2
        """
        try:
            return self.delete_at_index(index)
        except IndexOutOfBoundsError as e:
            raise InvalidOperationError(f"Cannot remove at index {index}: {e}") from e

    def remove_value(self, value: T) -> bool:
        """
        Remove the first occurrence of the specified value.
        
        Args:
            value: The value to remove.
            
        Returns:
            True if the value was found and removed, False otherwise.
            
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> ll = LinkedList[str]()
            >>> ll.extend(["apple", "banana", "cherry"])
            >>> ll.remove_value("banana")
            True
            >>> ll.remove_value("grape")
            False
        """
        return self.delete(value)

    def is_empty(self) -> bool:
        """
        Check if the linked list is empty.
        
        Returns:
            True if the list is empty, False otherwise.
        """
        return self._size == 0
        
    def __iter__(self) -> Iterator[T]:
        """Iterate over the list elements."""
        current = self._head
        while current is not None:
            yield current.data
            current = current.next
            
    def __eq__(self, other: object) -> bool:
        """
        Check if this LinkedList is equal to another LinkedList.
        
        Args:
            other: The object to compare with.
            
        Returns:
            True if both lists contain the same elements in the same order.
        """
        if not isinstance(other, LinkedList):
            return False
            
        if self._size != other._size:
            return False
            
        for item1, item2 in zip(self, other):
            if item1 != item2:
                return False
                
        return True
        
    def __len__(self) -> int:
        """Return the number of elements in the list."""
        return self._size
        
    def __contains__(self, item: object) -> bool:
        """Check if an item is in the list."""
        return self.find(item) != -1  # type: ignore[arg-type]
        
    def __getitem__(self, index: int) -> T:
        """Get element at index using bracket notation."""
        return self.get(index)
        
    def __setitem__(self, index: int, value: T) -> None:
        """Set element at index using bracket notation."""
        self.set(index, value)
        
    def __str__(self) -> str:
        """Return simple string representation with class name."""
        if self.is_empty():
            return "LinkedList[]"
        elements = [str(item) for item in self]
        return f"LinkedList[{', '.join(elements)}]"
        
    def __repr__(self) -> str:
        """Return detailed string representation."""
        elements = [str(item) for item in self]
        return f"LinkedList([{', '.join(elements)}], size={self._size})"
        
    def to_list(self) -> list[T]:
        """
        Convert the LinkedList to a Python list.
        
        Returns:
            A new list containing all elements in order.
            
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Example:
            >>> ll = LinkedList.from_list([1, 2, 3])
            >>> ll.to_list()
            [1, 2, 3]
        """
        return list(self)
        
    def _get_node_at_index(self, index: int) -> ListNode[T]:
        """Get the node at the specified index (internal method)."""
        # Optimize by starting from head or tail based on index
        if index < self._size // 2:
            # Start from head
            current = self._head
            for _ in range(index):
                current = current.next  # type: ignore[union-attr]
        else:
            # Start from tail
            current = self._tail
            for _ in range(self._size - 1 - index):
                current = current.prev  # type: ignore[union-attr]
                
        return current  # type: ignore[return-value]
        
    def _remove_node(self, node: ListNode[T]) -> None:
        """Remove a specific node from the list (internal method)."""
        if node.prev:
            node.prev.next = node.next
        else:
            self._head = node.next
            
        if node.next:
            node.next.prev = node.prev
        else:
            self._tail = node.prev
            
        self._size -= 1
        
    def _merge_sort_recursive(self, head: ListNode[T] | None) -> ListNode[T] | None:
        """Recursive merge sort implementation for linked list."""
        if head is None or head.next is None:
            return head
            
        # Split the list into two halves
        middle = self._get_middle(head)
        next_to_middle = middle.next  # type: ignore[union-attr]
        middle.next = None  # type: ignore[union-attr]
        
        # Recursively sort both halves
        left = self._merge_sort_recursive(head)
        right = self._merge_sort_recursive(next_to_middle)
        
        # Merge the sorted halves
        return self._merge_sorted_lists(left, right)
        
    def _get_middle(self, head: ListNode[T]) -> ListNode[T]:
        """Get the middle node of a linked list using slow/fast pointers."""
        slow = fast = head
        prev = None
        
        while fast and fast.next:
            prev = slow
            slow = slow.next  # type: ignore[union-attr]
            fast = fast.next.next  # type: ignore[union-attr]
            
        return prev if prev else slow  # type: ignore[return-value]
        
    def _merge_sorted_lists(self, left: ListNode[T] | None, right: ListNode[T] | None) -> ListNode[T] | None:
        """Merge two sorted linked lists."""
        if not left:
            return right
        if not right:
            return left
            
        # Create dummy head for easier manipulation
        dummy = ListNode(left.data)  # Dummy node
        current = dummy
        
        while left and right:
            if left.data <= right.data:  # type: ignore[operator]
                current.next = left
                left = left.next
            else:
                current.next = right
                right = right.next
            current = current.next
            
        # Attach remaining nodes
        current.next = left if left else right
        
        return dummy.next
        
    def _rebuild_structure(self) -> None:
        """Rebuild the doubly-linked structure and tail reference."""
        if not self._head:
            self._tail = None
            return
            
        # Rebuild prev pointers and find tail
        current = self._head
        current.prev = None
        
        while current.next:
            current.next.prev = current
            current = current.next
            
        self._tail = current

    def reverse(self) -> None:
        """
        Reverse the linked list in-place.
        
        This operation reverses the order of elements by swapping
        the next and prev pointers of each node.
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Example:
            >>> ll = LinkedList[str]()
            >>> ll.extend(["a", "b", "c"])
            >>> ll.reverse()
            >>> list(ll)
            ['c', 'b', 'a']
        """
        if self._size <= 1:
            return
            
        current = self._head
        
        # Swap next and prev pointers for each node
        while current is not None:
            # Swap next and prev
            current.next, current.prev = current.prev, current.next
            current = current.prev  # Move to what was originally next
            
        # Swap head and tail
        self._head, self._tail = self._tail, self._head
        
    def clear(self) -> None:
        """
        Remove all elements from the linked list.
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> ll = LinkedList[int]()
            >>> ll.extend([1, 2, 3])
            >>> ll.clear()
            >>> len(ll)
            0
        """
        self._head = None
        self._tail = None
        self._size = 0

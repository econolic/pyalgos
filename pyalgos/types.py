"""
Common type definitions and protocols used throughout the PyAlgos library.

This module defines type aliases, protocols, and generic types that ensure
consistency and type safety across all data structure implementations.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, TypeVar, runtime_checkable

# Generic type variables for reuse across the library
T = TypeVar("T")  # Generic type for data elements
K = TypeVar("K")  # Generic type for keys
V = TypeVar("V")  # Generic type for values

# Numeric types commonly used in algorithms
Number = int | float
Comparable = TypeVar("Comparable", bound="SupportsComparison")


@runtime_checkable
class SupportsComparison(Protocol):
    """Protocol for types that support comparison operations.
    
    This protocol defines the interface for types that can be compared
    using standard comparison operators. It's used as a bound for generic
    types in data structures that require ordering.
    """

    @abstractmethod
    def __lt__(self, other: Any) -> bool:
        """Less than comparison."""
        ...

    @abstractmethod
    def __le__(self, other: Any) -> bool:
        """Less than or equal comparison."""
        ...

    @abstractmethod
    def __gt__(self, other: Any) -> bool:
        """Greater than comparison."""
        ...

    @abstractmethod
    def __ge__(self, other: Any) -> bool:
        """Greater than or equal comparison."""
        ...

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        ...

    @abstractmethod
    def __ne__(self, other: Any) -> bool:
        """Inequality comparison."""
        ...


@runtime_checkable
class Container(Protocol):
    """Protocol for container data structures.
    
    This protocol defines the basic interface that all container data
    structures should implement, providing consistent methods for
    size checking and element membership testing.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of elements in the container."""
        ...

    @abstractmethod
    def __contains__(self, item: object) -> bool:
        """Check if an item is in the container."""
        ...

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if the container is empty."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all elements from the container."""
        ...


@runtime_checkable
class Stack(Protocol[T]):
    """Protocol defining the stack interface.
    
    A stack follows LIFO (Last In, First Out) ordering and supports
    push and pop operations at the top of the stack.
    """

    @abstractmethod
    def push(self, item: T) -> None:
        """Push an item onto the top of the stack."""
        ...

    @abstractmethod
    def pop(self) -> T:
        """Remove and return the top item from the stack."""
        ...

    @abstractmethod
    def peek(self) -> T:
        """Return the top item without removing it."""
        ...


@runtime_checkable
class Queue(Protocol[T]):
    """Protocol defining the queue interface.
    
    A queue follows FIFO (First In, First Out) ordering and supports
    enqueue at the rear and dequeue at the front.
    """

    @abstractmethod
    def enqueue(self, item: T) -> None:
        """Add an item to the rear of the queue."""
        ...

    @abstractmethod
    def dequeue(self) -> T:
        """Remove and return the front item from the queue."""
        ...

    @abstractmethod
    def peek(self) -> T:
        """Return the front item without removing it."""
        ...


@runtime_checkable
class Heap(Protocol[T]):
    """Protocol defining the heap interface.
    
    A heap is a binary tree that maintains the heap property:
    for min-heaps, each parent is smaller than its children.
    """

    @abstractmethod
    def insert(self, item: T) -> None:
        """Insert an item into the heap."""
        ...

    @abstractmethod
    def extract_min(self) -> T:
        """Remove and return the minimum item from the heap."""
        ...

    @abstractmethod
    def peek_min(self) -> T:
        """Return the minimum item without removing it."""
        ...

    @abstractmethod
    def heapify(self) -> None:
        """Restore the heap property."""
        ...


# Type aliases for common use cases
SizeType = int  # For sizes and indices
IndexType = int  # For array/list indices
HashType = int  # For hash values

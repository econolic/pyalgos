"""
Common exceptions used throughout the PyAlgos library.

This module defines custom exception classes that provide meaningful error messages
and maintain consistency across all data structure implementations.
"""

from __future__ import annotations


class PyAlgosError(Exception):
    """Base exception class for all PyAlgos errors.
    
    This serves as the parent class for all custom exceptions in the library,
    allowing users to catch all PyAlgos-specific errors with a single except clause.
    """

    pass


class EmptyStructureError(PyAlgosError):
    """Raised when attempting to access elements from an empty data structure.
    
    This exception is thrown when operations like pop(), peek(), or dequeue()
    are called on empty stacks, queues, heaps, or other structures.
    """

    def __init__(self, structure_name: str) -> None:
        """Initialize the exception with the name of the empty structure.
        
        Args:
            structure_name: The name of the data structure that is empty.
        """
        super().__init__(f"Cannot perform operation on empty {structure_name}")


class IndexOutOfBoundsError(PyAlgosError):
    """Raised when an index is outside the valid range for a data structure.
    
    This exception is thrown when accessing elements with invalid indices
    in arrays, lists, or other indexed structures.
    """

    def __init__(self, index: int, size: int, structure_name: str = "structure") -> None:
        """Initialize the exception with index information.
        
        Args:
            index: The invalid index that was accessed.
            size: The current size of the structure.
            structure_name: The name of the data structure.
        """
        super().__init__(
            f"Index {index} is out of bounds for {structure_name} of size {size}"
        )


class InvalidOperationError(PyAlgosError):
    """Raised when an invalid operation is attempted on a data structure.
    
    This exception is thrown when operations that violate the invariants
    or constraints of a data structure are attempted.
    """

    def __init__(self, message: str) -> None:
        """Initialize the exception with a descriptive message.
        
        Args:
            message: A description of the invalid operation.
        """
        super().__init__(message)


class DuplicateKeyError(PyAlgosError):
    """Raised when attempting to insert a duplicate key where not allowed.
    
    This exception is thrown by data structures that require unique keys
    when a duplicate key insertion is attempted.
    """

    def __init__(self, key: object, structure_name: str) -> None:
        """Initialize the exception with key information.
        
        Args:
            key: The duplicate key that was rejected.
            structure_name: The name of the data structure.
        """
        super().__init__(f"Key '{key}' already exists in {structure_name}")


class KeyNotFoundError(PyAlgosError):
    """Raised when a requested key is not found in a data structure.
    
    This exception is thrown when search, delete, or update operations
    are performed with keys that don't exist in the structure.
    """

    def __init__(self, key: object, structure_name: str) -> None:
        """Initialize the exception with key information.
        
        Args:
            key: The key that was not found.
            structure_name: The name of the data structure.
        """
        super().__init__(f"Key '{key}' not found in {structure_name}")

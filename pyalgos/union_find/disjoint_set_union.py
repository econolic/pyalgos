"""
Disjoint Set Union (Union-Find) implementation with path compression and union by size.

This module provides a Disjoint Set Union data structure for
dynamic connectivity problems. The implementation includes path compression
and union by size optimizations for near-constant time operations.

Key Features:
    - Union by size for balanced tree structure
    - Path compression for efficient find operations
    - Nearly O(α(n)) amortized time complexity where α is inverse Ackermann
    - Component connectivity queries
    - Component size tracking
    - Input validation and error handling
    - Support for both integer and generic element types

Example:
    >>> from pyalgos.union_find.disjoint_set_union import DisjointSetUnion
    >>> dsu = DisjointSetUnion(5)  # Create DSU for elements 0-4
    >>> dsu.union(0, 1)
    >>> dsu.union(2, 3)
    >>> print(dsu.connected(0, 1))  # True
    >>> print(dsu.connected(0, 2))  # False
    >>> print(dsu.num_components())  # 3 (components: {0,1}, {2,3}, {4})
"""

from __future__ import annotations

from typing import Dict, Generic, List, Set, TypeVar

from pyalgos.exceptions import IndexOutOfBoundsError, InvalidOperationError
from pyalgos.types import SizeType

T = TypeVar("T")


class DisjointSetUnion:
    """
    A Disjoint Set Union (Union-Find) data structure with optimizations.
    
    This implementation uses path compression and union by size to achieve
    nearly O(α(n)) amortized time complexity for union and find operations,
    where α is the inverse Ackermann function.
    
    The DSU maintains disjoint sets and supports efficient operations for:
    - Finding which set an element belongs to
    - Uniting two sets
    - Checking if two elements are in the same set
    - Tracking the number of disjoint components
    
    Attributes:
        _parent: Parent array for the union-find structure.
        _size: Size of each component (root only).
        _num_elements: Total number of elements.
        _num_components: Current number of disjoint components.
        
    Time Complexities:
        - find(x): O(α(n)) amortized with path compression
        - union(x, y): O(α(n)) amortized
        - connected(x, y): O(α(n)) amortized
        - component_size(x): O(α(n)) amortized
        
    Space Complexity: O(n) where n is the number of elements
    """
    
    def __init__(self, n: int) -> None:
        """
        Initialize DSU for n elements (0 to n-1).
        
        Args:
            n: Number of elements in the DSU (must be positive).
            
        Raises:
            InvalidOperationError: If n is not positive.
            
        Example:
            >>> dsu = DisjointSetUnion(5)  # Elements 0, 1, 2, 3, 4
            >>> dsu.num_components()
            5
        """
        if not isinstance(n, int) or n <= 0:
            raise InvalidOperationError(f"Number of elements must be positive integer, got {n}")
            
        self._num_elements = n
        self._num_components = n
        
        # Initialize each element as its own parent (separate component)
        self._parent = list(range(n))
        
        # Initialize size of each component as 1
        self._size = [1] * n
        
    def find(self, x: int) -> int:
        """
        Find the root (representative) of the set containing x.
        
        Uses path compression optimization where all nodes on the path
        to the root are made direct children of the root.
        
        Args:
            x: Element to find the root for.
            
        Returns:
            The root of the set containing x.
            
        Raises:
            IndexOutOfBoundsError: If x is not a valid element.
            
        Time Complexity: O(α(n)) amortized with path compression.
        Space Complexity: O(1).
        
        Example:
            >>> dsu = DisjointSetUnion(5)
            >>> dsu.union(0, 1)
            >>> dsu.union(1, 2)
            >>> root = dsu.find(2)  # Returns the root of component {0,1,2}
        """
        self._validate_element(x)
        
        # Path compression: make x point directly to root
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
            
        return self._parent[x]
        
    def union(self, x: int, y: int) -> bool:
        """
        Union the sets containing x and y.
        
        Uses union by size optimization where the smaller tree is attached
        to the root of the larger tree to maintain balance.
        
        Args:
            x: First element.
            y: Second element.
            
        Returns:
            True if union was performed (elements were in different sets),
            False if elements were already in the same set.
            
        Raises:
            IndexOutOfBoundsError: If x or y are not valid elements.
            
        Time Complexity: O(α(n)) amortized.
        Space Complexity: O(1).
        
        Example:
            >>> dsu = DisjointSetUnion(4)
            >>> dsu.union(0, 1)  # True (different sets)
            >>> dsu.union(0, 1)  # False (already in same set)
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        # Already in same set
        if root_x == root_y:
            return False
            
        # Union by size: attach smaller tree to larger tree
        if self._size[root_x] < self._size[root_y]:
            root_x, root_y = root_y, root_x
            
        # Attach root_y to root_x
        self._parent[root_y] = root_x
        self._size[root_x] += self._size[root_y]
        
        # Decrease number of components
        self._num_components -= 1
        
        return True
        
    def connected(self, x: int, y: int) -> bool:
        """
        Check if x and y are in the same connected component.
        
        Args:
            x: First element.
            y: Second element.
            
        Returns:
            True if x and y are connected, False otherwise.
            
        Raises:
            IndexOutOfBoundsError: If x or y are not valid elements.
            
        Time Complexity: O(α(n)) amortized.
        Space Complexity: O(1).
        
        Example:
            >>> dsu = DisjointSetUnion(4)
            >>> dsu.union(0, 1)
            >>> dsu.connected(0, 1)  # True
            >>> dsu.connected(0, 2)  # False
        """
        return self.find(x) == self.find(y)
        
    def component_size(self, x: int) -> int:
        """
        Get the size of the component containing x.
        
        Args:
            x: Element to get component size for.
            
        Returns:
            The size of the component containing x.
            
        Raises:
            IndexOutOfBoundsError: If x is not a valid element.
            
        Time Complexity: O(α(n)) amortized.
        Space Complexity: O(1).
        
        Example:
            >>> dsu = DisjointSetUnion(5)
            >>> dsu.union(0, 1)
            >>> dsu.union(1, 2)
            >>> dsu.component_size(0)  # 3 (component {0,1,2})
        """
        root = self.find(x)
        return self._size[root]
        
    def num_components(self) -> int:
        """
        Get the current number of disjoint components.
        
        Returns:
            The number of disjoint components.
            
        Time Complexity: O(1).
        Space Complexity: O(1).
        
        Example:
            >>> dsu = DisjointSetUnion(5)  # 5 components initially
            >>> dsu.union(0, 1)           # 4 components
            >>> dsu.union(2, 3)           # 3 components
            >>> dsu.num_components()      # 3
        """
        return self._num_components
        
    def get_components(self) -> List[List[int]]:
        """
        Get all components as lists of elements.
        
        Returns:
            A list where each inner list represents a connected component.
            
        Time Complexity: O(n × α(n)).
        Space Complexity: O(n).
        
        Example:
            >>> dsu = DisjointSetUnion(5)
            >>> dsu.union(0, 1)
            >>> dsu.union(2, 3)
            >>> components = dsu.get_components()
            >>> # Returns something like: [[0, 1], [2, 3], [4]]
        """
        components: Dict[int, List[int]] = {}
        
        for i in range(self._num_elements):
            root = self.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
            
        return list(components.values())
        
    def get_component_roots(self) -> Set[int]:
        """
        Get all component root elements.
        
        Returns:
            A set of root elements representing each component.
            
        Time Complexity: O(n × α(n)).
        Space Complexity: O(n).
        
        Example:
            >>> dsu = DisjointSetUnion(5)
            >>> dsu.union(0, 1)
            >>> roots = dsu.get_component_roots()
            >>> len(roots)  # Number of components
        """
        roots = set()
        for i in range(self._num_elements):
            roots.add(self.find(i))
        return roots
        
    def reset(self) -> None:
        """
        Reset the DSU to initial state where each element is its own component.
        
        Time Complexity: O(n).
        Space Complexity: O(1).
        
        Example:
            >>> dsu = DisjointSetUnion(3)
            >>> dsu.union(0, 1)
            >>> dsu.num_components()  # 2
            >>> dsu.reset()
            >>> dsu.num_components()  # 3
        """
        self._num_components = self._num_elements
        self._parent = list(range(self._num_elements))
        self._size = [1] * self._num_elements
        
    def is_fully_connected(self) -> bool:
        """
        Check if all elements are in a single component.
        
        Returns:
            True if there's only one component, False otherwise.
            
        Time Complexity: O(1).
        Space Complexity: O(1).
        
        Example:
            >>> dsu = DisjointSetUnion(3)
            >>> dsu.is_fully_connected()  # False
            >>> dsu.union(0, 1)
            >>> dsu.union(1, 2)
            >>> dsu.is_fully_connected()  # True
        """
        return self._num_components == 1
        
    def largest_component_size(self) -> int:
        """
        Get the size of the largest component.
        
        Returns:
            The size of the largest component.
            
        Time Complexity: O(n).
        Space Complexity: O(1).
        
        Example:
            >>> dsu = DisjointSetUnion(5)
            >>> dsu.union(0, 1)
            >>> dsu.union(1, 2)  # Component of size 3
            >>> dsu.largest_component_size()  # 3
        """
        max_size = 0
        visited_roots = set()
        
        for i in range(self._num_elements):
            root = self.find(i)
            if root not in visited_roots:
                visited_roots.add(root)
                max_size = max(max_size, self._size[root])
                
        return max_size
        
    def _validate_element(self, x: int) -> None:
        """
        Validate that x is a valid element.
        
        Args:
            x: Element to validate.
            
        Raises:
            IndexOutOfBoundsError: If x is not a valid element.        """
        if not isinstance(x, int):
            raise IndexOutOfBoundsError(-1, self._num_elements, "DisjointSetUnion")
            
        if not (0 <= x < self._num_elements):
            raise IndexOutOfBoundsError(x, self._num_elements, "DisjointSetUnion")
            
    def __len__(self) -> int:
        """Return the number of elements in the DSU."""
        return self._num_elements
        
    def __contains__(self, x: int) -> bool:
        """Check if x is a valid element in the DSU."""
        return isinstance(x, int) and 0 <= x < self._num_elements
        
    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"DisjointSetUnion(n={self._num_elements}, components={self._num_components})"
        
    def __str__(self) -> str:
        """Return simple string representation."""
        components = self.get_components()
        if len(components) <= 3:
            return f"DSU{components}"
        else:
            return f"DSU[{len(components)} components, largest={self.largest_component_size()}]"


class GenericDisjointSetUnion(Generic[T]):
    """
    A generic Disjoint Set Union that works with any hashable type.
    
    This implementation wraps the integer-based DSU to support arbitrary
    hashable elements by maintaining a mapping between elements and indices.
    
    Example:
        >>> dsu = GenericDisjointSetUnion()
        >>> dsu.add_element("Alice")
        >>> dsu.add_element("Bob")
        >>> dsu.add_element("Charlie")
        >>> dsu.union("Alice", "Bob")
        >>> dsu.connected("Alice", "Bob")  # True
    """
    
    def __init__(self) -> None:
        """Initialize an empty generic DSU."""
        self._element_to_index: Dict[T, int] = {}
        self._index_to_element: Dict[int, T] = {}
        self._dsu = DisjointSetUnion(0)  # Will be recreated when elements are added
        self._next_index = 0
        
    def add_element(self, element: T) -> None:
        """
        Add an element to the DSU.
        
        Args:
            element: The element to add (must be hashable).
            
        Raises:
            InvalidOperationError: If element already exists.
        """
        if element in self._element_to_index:
            raise InvalidOperationError(f"Element {element} already exists")
            
        self._element_to_index[element] = self._next_index
        self._index_to_element[self._next_index] = element
        self._next_index += 1
        
        # Recreate DSU with new size
        old_components = []
        if len(self._dsu) > 0:
            old_components = self._dsu.get_components()
            
        self._dsu = DisjointSetUnion(self._next_index)
        
        # Restore old connections
        for component in old_components:
            if len(component) > 1:
                for i in range(1, len(component)):
                    self._dsu.union(component[0], component[i])
                    
    def union(self, x: T, y: T) -> bool:
        """Union the sets containing x and y."""
        x_idx = self._get_index(x)
        y_idx = self._get_index(y)
        return self._dsu.union(x_idx, y_idx)
        
    def connected(self, x: T, y: T) -> bool:
        """Check if x and y are connected."""
        x_idx = self._get_index(x)
        y_idx = self._get_index(y)
        return self._dsu.connected(x_idx, y_idx)
        
    def component_size(self, x: T) -> int:
        """Get the size of the component containing x."""
        x_idx = self._get_index(x)
        return self._dsu.component_size(x_idx)
        
    def num_components(self) -> int:
        """Get the number of components."""
        return self._dsu.num_components()
        
    def get_components(self) -> List[List[T]]:
        """Get all components as lists of elements."""
        index_components = self._dsu.get_components()
        return [
            [self._index_to_element[idx] for idx in component]
            for component in index_components
        ]
        
    def _get_index(self, element: T) -> int:
        """Get the index for an element."""
        if element not in self._element_to_index:
            raise InvalidOperationError(f"Element {element} not found")
        return self._element_to_index[element]
        
    def __len__(self) -> int:
        """Return the number of elements."""
        return len(self._element_to_index)
        
    def __contains__(self, element: T) -> bool:
        """Check if element is in the DSU."""
        return element in self._element_to_index
        
    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"GenericDisjointSetUnion(elements={len(self._element_to_index)}, components={self.num_components()})"

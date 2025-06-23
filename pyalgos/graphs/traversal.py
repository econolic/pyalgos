"""
Graph traversal algorithms implementation.

This module provides efficient implementations of fundamental graph traversal
algorithms including Breadth-First Search (BFS) and Depth-First Search (DFS).
The algorithms support both recursive and iterative approaches.

Key Features:
    - BFS and DFS traversal algorithms
    - Path finding and connectivity checking
    - Component detection in graphs
    - Flexible visitor pattern support
    - Comprehensive input validation and error handling
    - Generic vertex type support

Example:
    >>> from pyalgos.graphs.graph import Graph
    >>> from pyalgos.graphs.traversal import BFS, DFS
    >>> g = Graph[str](directed=False)
    >>> g.add_vertex("A")
    >>> g.add_vertex("B")
    >>> g.add_edge("A", "B")
    >>> bfs = BFS(g)
    >>> path = bfs.find_path("A", "B")  # ["A", "B"]
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Dict, Generic, List, Optional, Set, TypeVar

from pyalgos.exceptions import InvalidOperationError
from pyalgos.graphs.graph import Graph

T = TypeVar("T")


class BFS(Generic[T]):
    """
    Breadth-First Search implementation for graphs.
    
    BFS explores vertices in order of their distance from the start vertex,
    visiting all neighbors at the current depth before moving to vertices
    at the next depth level.
    
    Time Complexity: O(V + E) where V is vertices, E is edges
    Space Complexity: O(V)
    
    Attributes:
        _graph: The graph to traverse.
    """
    
    def __init__(self, graph: Graph[T]) -> None:
        """
        Initialize BFS with a graph.
        
        Args:
            graph: The graph to perform BFS on.
        """
        self._graph = graph
        
    def traverse(
        self, 
        start: T, 
        visitor: Optional[Callable[[T], None]] = None
    ) -> List[T]:
        """
        Perform BFS traversal starting from a vertex.
        
        Args:
            start: Starting vertex for traversal.
            visitor: Optional function to call on each visited vertex.
            
        Returns:
            List of vertices in BFS order.
            
        Raises:
            InvalidOperationError: If start vertex doesn't exist.
            
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Example:
            >>> g = Graph[str](directed=False)
            >>> g.add_vertex("A")
            >>> g.add_vertex("B")
            >>> g.add_edge("A", "B")
            >>> bfs = BFS(g)
            >>> result = bfs.traverse("A")  # ["A", "B"]
        """
        if not self._graph.has_vertex(start):
            raise InvalidOperationError(f"Start vertex {start} does not exist")
            
        visited: Set[T] = set()
        queue: deque[T] = deque([start])
        result: List[T] = []
        
        visited.add(start)
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            if visitor:
                visitor(current)
                
            # Add unvisited neighbors to queue
            for neighbor in self._graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        return result
        
    def find_path(self, start: T, target: T) -> Optional[List[T]]:
        """
        Find a path from start to target using BFS.
        
        Args:
            start: Starting vertex.
            target: Target vertex.
            
        Returns:
            Path from start to target, or None if no path exists.
            
        Raises:
            InvalidOperationError: If vertices don't exist.
            
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Example:
            >>> g = Graph[str](directed=False)
            >>> g.add_vertex("A")
            >>> g.add_vertex("B")
            >>> g.add_vertex("C")
            >>> g.add_edge("A", "B")
            >>> g.add_edge("B", "C")
            >>> bfs = BFS(g)
            >>> path = bfs.find_path("A", "C")  # ["A", "B", "C"]
        """
        if not self._graph.has_vertex(start):
            raise InvalidOperationError(f"Start vertex {start} does not exist")
        if not self._graph.has_vertex(target):
            raise InvalidOperationError(f"Target vertex {target} does not exist")
            
        if start == target:
            return [start]
            
        visited: Set[T] = set()
        queue: deque[T] = deque([start])
        parent: Dict[T, Optional[T]] = {start: None}
        
        visited.add(start)
        
        while queue:
            current = queue.popleft()
            
            if current == target:
                # Reconstruct path
                path = []
                node = target
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return path[::-1]  # Reverse to get start->target
                
            # Add unvisited neighbors to queue
            for neighbor in self._graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
                    
        return None  # No path found
        
    def shortest_distance(self, start: T, target: T) -> Optional[int]:
        """
        Find the shortest distance (number of edges) from start to target.
        
        Args:
            start: Starting vertex.
            target: Target vertex.
            
        Returns:
            Shortest distance, or None if no path exists.
            
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        path = self.find_path(start, target)
        return len(path) - 1 if path else None
        
    def is_connected(self, start: T, target: T) -> bool:
        """
        Check if two vertices are connected.
        
        Args:
            start: Starting vertex.
            target: Target vertex.
            
        Returns:
            True if vertices are connected, False otherwise.
        """
        return self.find_path(start, target) is not None
        
    def get_connected_component(self, start: T) -> Set[T]:
        """
        Get all vertices in the same connected component as start.
        
        Args:
            start: Starting vertex.
            
        Returns:
            Set of vertices in the same component.
            
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        if not self._graph.has_vertex(start):
            raise InvalidOperationError(f"Start vertex {start} does not exist")
            
        visited: Set[T] = set()
        queue: deque[T] = deque([start])
        component: Set[T] = set()
        
        visited.add(start)
        
        while queue:
            current = queue.popleft()
            component.add(current)
            
            for neighbor in self._graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        return component


class DFS(Generic[T]):
    """
    Depth-First Search implementation for graphs.
    
    DFS explores as far as possible along each branch before backtracking,
    visiting vertices in a depth-first manner.
    
    Time Complexity: O(V + E) where V is vertices, E is edges
    Space Complexity: O(V)
    
    Attributes:
        _graph: The graph to traverse.
    """
    
    def __init__(self, graph: Graph[T]) -> None:
        """
        Initialize DFS with a graph.
        
        Args:
            graph: The graph to perform DFS on.
        """
        self._graph = graph
        
    def traverse(
        self, 
        start: T, 
        visitor: Optional[Callable[[T], None]] = None,
        recursive: bool = True
    ) -> List[T]:
        """
        Perform DFS traversal starting from a vertex.
        
        Args:
            start: Starting vertex for traversal.
            visitor: Optional function to call on each visited vertex.
            recursive: Whether to use recursive implementation.
            
        Returns:
            List of vertices in DFS order.
            
        Raises:
            InvalidOperationError: If start vertex doesn't exist.
            
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Example:
            >>> g = Graph[str](directed=False)
            >>> g.add_vertex("A")
            >>> g.add_vertex("B")
            >>> g.add_edge("A", "B")
            >>> dfs = DFS(g)
            >>> result = dfs.traverse("A")  # ["A", "B"]
        """
        if not self._graph.has_vertex(start):
            raise InvalidOperationError(f"Start vertex {start} does not exist")
            
        if recursive:
            return self._traverse_recursive(start, visitor)
        else:
            return self._traverse_iterative(start, visitor)
            
    def _traverse_recursive(
        self, 
        start: T, 
        visitor: Optional[Callable[[T], None]] = None
    ) -> List[T]:
        """Recursive DFS implementation."""
        visited: Set[T] = set()
        result: List[T] = []
        
        def _dfs_helper(vertex: T) -> None:
            visited.add(vertex)
            result.append(vertex)
            
            if visitor:
                visitor(vertex)
                
            for neighbor in self._graph.get_neighbors(vertex):
                if neighbor not in visited:
                    _dfs_helper(neighbor)
                    
        _dfs_helper(start)
        return result
        
    def _traverse_iterative(
        self, 
        start: T, 
        visitor: Optional[Callable[[T], None]] = None
    ) -> List[T]:
        """Iterative DFS implementation using a stack."""
        visited: Set[T] = set()
        stack: List[T] = [start]
        result: List[T] = []
        
        while stack:
            current = stack.pop()
            
            if current not in visited:
                visited.add(current)
                result.append(current)
                
                if visitor:
                    visitor(current)
                    
                # Add neighbors to stack (in reverse order for consistent ordering)
                neighbors = self._graph.get_neighbors(current)
                for neighbor in reversed(neighbors):
                    if neighbor not in visited:
                        stack.append(neighbor)
                        
        return result
        
    def find_path(self, start: T, target: T) -> Optional[List[T]]:
        """
        Find a path from start to target using DFS.
        
        Args:
            start: Starting vertex.
            target: Target vertex.
            
        Returns:
            Path from start to target, or None if no path exists.
            
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        if not self._graph.has_vertex(start):
            raise InvalidOperationError(f"Start vertex {start} does not exist")
        if not self._graph.has_vertex(target):
            raise InvalidOperationError(f"Target vertex {target} does not exist")
            
        if start == target:
            return [start]
            
        visited: Set[T] = set()
        path: List[T] = []
        
        def _dfs_path_helper(vertex: T) -> bool:
            visited.add(vertex)
            path.append(vertex)
            
            if vertex == target:
                return True
                
            for neighbor in self._graph.get_neighbors(vertex):
                if neighbor not in visited:
                    if _dfs_path_helper(neighbor):
                        return True
                        
            path.pop()  # Backtrack
            return False
            
        if _dfs_path_helper(start):
            return path
        return None
        
    def has_cycle(self) -> bool:
        """
        Detect if the graph has a cycle using DFS.
        
        Returns:
            True if graph has a cycle, False otherwise.
            
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        if self._graph.is_directed():
            return self._has_cycle_directed()
        else:
            return self._has_cycle_undirected()
            
    def _has_cycle_directed(self) -> bool:
        """Cycle detection for directed graphs using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[T, int] = {}
        
        # Initialize all vertices as WHITE
        for vertex in self._graph.get_vertices():
            color[vertex] = WHITE
            
        def _has_cycle_helper(vertex: T) -> bool:
            color[vertex] = GRAY
            
            for neighbor in self._graph.get_neighbors(vertex):
                if color[neighbor] == GRAY:  # Back edge found
                    return True
                elif color[neighbor] == WHITE and _has_cycle_helper(neighbor):
                    return True
                    
            color[vertex] = BLACK
            return False
            
        for vertex in self._graph.get_vertices():
            if color[vertex] == WHITE:
                if _has_cycle_helper(vertex):
                    return True
                    
        return False
        
    def _has_cycle_undirected(self) -> bool:
        """Cycle detection for undirected graphs using DFS."""
        visited: Set[T] = set()
        
        def _has_cycle_helper(vertex: T, parent: Optional[T]) -> bool:
            visited.add(vertex)
            
            for neighbor in self._graph.get_neighbors(vertex):
                if neighbor not in visited:
                    if _has_cycle_helper(neighbor, vertex):
                        return True
                elif neighbor != parent:  # Back edge to non-parent
                    return True
                    
            return False
            
        for vertex in self._graph.get_vertices():
            if vertex not in visited:
                if _has_cycle_helper(vertex, None):
                    return True
                    
        return False
        
    def topological_sort(self) -> Optional[List[T]]:
        """
        Perform topological sort on a directed acyclic graph (DAG).
        
        Returns:
            Topologically sorted list of vertices, or None if graph has cycles.
            
        Raises:
            InvalidOperationError: If graph is not directed.
            
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        if not self._graph.is_directed():
            raise InvalidOperationError("Topological sort is only defined for directed graphs")
            
        if self.has_cycle():
            return None  # Cannot topologically sort a graph with cycles
            
        visited: Set[T] = set()
        stack: List[T] = []
        
        def _topological_helper(vertex: T) -> None:
            visited.add(vertex)
            
            for neighbor in self._graph.get_neighbors(vertex):
                if neighbor not in visited:
                    _topological_helper(neighbor)
                    
            stack.append(vertex)
            
        for vertex in self._graph.get_vertices():
            if vertex not in visited:
                _topological_helper(vertex)
                
        return stack[::-1]  # Reverse to get correct order

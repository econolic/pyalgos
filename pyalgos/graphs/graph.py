"""
Graph implementation with adjacency list representation.

This module provides a Graph data structure supporting both
directed and undirected graphs. The implementation uses adjacency list
representation for memory efficiency and includes graph operations.

Key Features:
    - Adjacency list representation for space efficiency
    - Support for directed and undirected graphs
    - Weighted and unweighted edges
    - Vertex and edge addition/removal operations
    - Neighbor queries and graph traversal support
    - Input validation and error handling
    - Generic vertex type support

Example:
    >>> from pyalgos.graphs.graph import Graph
    >>> g = Graph[str](directed=False)
    >>> g.add_vertex("A")
    >>> g.add_vertex("B")
    >>> g.add_edge("A", "B")
    >>> print(g.get_neighbors("A"))  # ["B"]
    >>> print(g.has_edge("A", "B"))  # True
"""

from __future__ import annotations

from typing import Dict, Generic, List, Optional, Set, TypeVar, Union

from pyalgos.exceptions import InvalidOperationError
from pyalgos.types import SizeType

T = TypeVar("T")


class Graph(Generic[T]):
    """
    A graph data structure using adjacency list representation.
    
    This implementation supports both directed and undirected graphs with
    optional edge weights. The adjacency list representation provides
    efficient space usage and fast neighbor queries.
    
    Attributes:
        _adjacency_list: Dictionary mapping vertices to their neighbors.
        _directed: Whether the graph is directed.
        _weighted: Whether the graph supports edge weights.
        _num_edges: Current number of edges in the graph.
        
    Time Complexities:
        - add_vertex(v): O(1)
        - add_edge(u, v): O(1) 
        - remove_vertex(v): O(V + E) where V is vertices, E is edges
        - remove_edge(u, v): O(1)
        - has_vertex(v): O(1)
        - has_edge(u, v): O(degree(u))
        - get_neighbors(v): O(1)
        
    Space Complexity: O(V + E) where V is vertices, E is edges
    """
    
    def __init__(self, directed: bool = True, weighted: bool = False) -> None:
        """
        Initialize a new graph.
        
        Args:
            directed: Whether the graph is directed. Defaults to True.
            weighted: Whether edges have weights. Defaults to False.
            
        Example:
            >>> g_directed = Graph[int](directed=True)
            >>> g_undirected = Graph[str](directed=False)
            >>> g_weighted = Graph[int](weighted=True)
        """
        self._adjacency_list: Dict[T, Dict[T, Union[bool, float]]] = {}
        self._directed = directed
        self._weighted = weighted
        self._num_edges = 0
        
    def add_vertex(self, vertex: T) -> None:
        """
        Add a vertex to the graph.
        
        Args:
            vertex: The vertex to add.
            
        Raises:
            InvalidOperationError: If vertex already exists.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> g = Graph[str]()
            >>> g.add_vertex("A")
            >>> g.add_vertex("B")
        """
        if vertex in self._adjacency_list:
            raise InvalidOperationError(f"Vertex {vertex} already exists")
            
        self._adjacency_list[vertex] = {}
        
    def remove_vertex(self, vertex: T) -> None:
        """
        Remove a vertex and all its incident edges from the graph.
        
        Args:
            vertex: The vertex to remove.
            
        Raises:
            InvalidOperationError: If vertex does not exist.
            
        Time Complexity: O(V + E) where V is vertices, E is edges
        Space Complexity: O(1)
        
        Example:
            >>> g = Graph[str]()
            >>> g.add_vertex("A")
            >>> g.add_vertex("B")
            >>> g.add_edge("A", "B")
            >>> g.remove_vertex("A")
        """
        if vertex not in self._adjacency_list:
            raise InvalidOperationError(f"Vertex {vertex} does not exist")
            
        # Remove all edges to this vertex
        edges_to_remove = 0
        for v in self._adjacency_list:
            if vertex in self._adjacency_list[v]:
                del self._adjacency_list[v][vertex]
                edges_to_remove += 1
                
        # Remove all edges from this vertex
        edges_to_remove += len(self._adjacency_list[vertex])
        
        # For undirected graphs, we've double-counted
        if not self._directed:
            edges_to_remove //= 2
            
        self._num_edges -= edges_to_remove
        
        # Remove the vertex itself
        del self._adjacency_list[vertex]
        
    def add_edge(self, source: T, target: T, weight: float = 1.0) -> None:
        """
        Add an edge between source and target vertices.
        
        Args:
            source: Source vertex.
            target: Target vertex.
            weight: Edge weight (only used if graph is weighted).
            
        Raises:
            InvalidOperationError: If vertices don't exist or edge already exists.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> g = Graph[str](weighted=True)
            >>> g.add_vertex("A")
            >>> g.add_vertex("B")
            >>> g.add_edge("A", "B", 5.0)
        """
        if source not in self._adjacency_list:
            raise InvalidOperationError(f"Source vertex {source} does not exist")
        if target not in self._adjacency_list:
            raise InvalidOperationError(f"Target vertex {target} does not exist")
            
        if target in self._adjacency_list[source]:
            raise InvalidOperationError(f"Edge from {source} to {target} already exists")
            
        # Add edge
        edge_value = weight if self._weighted else True
        self._adjacency_list[source][target] = edge_value
        
        # For undirected graphs, add reverse edge
        if not self._directed:
            self._adjacency_list[target][source] = edge_value
            
        self._num_edges += 1
        
    def remove_edge(self, source: T, target: T) -> None:
        """
        Remove an edge between source and target vertices.
        
        Args:
            source: Source vertex.
            target: Target vertex.
            
        Raises:
            InvalidOperationError: If vertices don't exist or edge doesn't exist.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Example:
            >>> g = Graph[str]()
            >>> g.add_vertex("A")
            >>> g.add_vertex("B")
            >>> g.add_edge("A", "B")
            >>> g.remove_edge("A", "B")
        """
        if source not in self._adjacency_list:
            raise InvalidOperationError(f"Source vertex {source} does not exist")
        if target not in self._adjacency_list:
            raise InvalidOperationError(f"Target vertex {target} does not exist")
            
        if target not in self._adjacency_list[source]:
            raise InvalidOperationError(f"Edge from {source} to {target} does not exist")
            
        # Remove edge
        del self._adjacency_list[source][target]
        
        # For undirected graphs, remove reverse edge
        if not self._directed:
            del self._adjacency_list[target][source]
            
        self._num_edges -= 1
        
    def has_vertex(self, vertex: T) -> bool:
        """
        Check if a vertex exists in the graph.
        
        Args:
            vertex: The vertex to check.
            
        Returns:
            True if vertex exists, False otherwise.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return vertex in self._adjacency_list
        
    def has_edge(self, source: T, target: T) -> bool:
        """
        Check if an edge exists between source and target vertices.
        
        Args:
            source: Source vertex.
            target: Target vertex.
            
        Returns:
            True if edge exists, False otherwise.
            
        Time Complexity: O(1) average case
        Space Complexity: O(1)
        """
        if source not in self._adjacency_list:
            return False
        return target in self._adjacency_list[source]
        
    def get_neighbors(self, vertex: T) -> List[T]:
        """
        Get all neighbors of a vertex.
        
        Args:
            vertex: The vertex to get neighbors for.
            
        Returns:
            List of neighboring vertices.
            
        Raises:
            InvalidOperationError: If vertex does not exist.
            
        Time Complexity: O(degree(vertex))
        Space Complexity: O(degree(vertex))
        
        Example:
            >>> g = Graph[str]()
            >>> g.add_vertex("A")
            >>> g.add_vertex("B")
            >>> g.add_edge("A", "B")
            >>> g.get_neighbors("A")  # ["B"]
        """
        if vertex not in self._adjacency_list:
            raise InvalidOperationError(f"Vertex {vertex} does not exist")
            
        return list(self._adjacency_list[vertex].keys())
        
    def get_edge_weight(self, source: T, target: T) -> float:
        """
        Get the weight of an edge.
        
        Args:
            source: Source vertex.
            target: Target vertex.
            
        Returns:
            The edge weight (1.0 for unweighted graphs).
            
        Raises:
            InvalidOperationError: If edge does not exist.
        """
        if not self.has_edge(source, target):
            raise InvalidOperationError(f"Edge from {source} to {target} does not exist")
            
        edge_value = self._adjacency_list[source][target]
        return float(edge_value) if self._weighted else 1.0
        
    def get_vertices(self) -> List[T]:
        """
        Get all vertices in the graph.
        
        Returns:
            List of all vertices.
            
        Time Complexity: O(V) where V is number of vertices
        Space Complexity: O(V)
        """
        return list(self._adjacency_list.keys())
        
    def get_edges(self) -> List[tuple[T, T]]:
        """
        Get all edges in the graph.
        
        Returns:
            List of edge tuples (source, target).
            
        Time Complexity: O(V + E) where V is vertices, E is edges
        Space Complexity: O(E)
        """
        edges = []
        visited_edges = set()
        
        for source in self._adjacency_list:
            for target in self._adjacency_list[source]:
                edge = (source, target)
                # For undirected graphs, avoid duplicates
                if not self._directed:
                    reverse_edge = (target, source)
                    if reverse_edge in visited_edges:
                        continue
                edges.append(edge)
                visited_edges.add(edge)
                
        return edges
        
    def num_vertices(self) -> SizeType:
        """
        Get the number of vertices in the graph.
        
        Returns:
            Number of vertices.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return len(self._adjacency_list)
        
    def num_edges(self) -> SizeType:
        """
        Get the number of edges in the graph.
        
        Returns:
            Number of edges.
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self._num_edges
        
    def is_directed(self) -> bool:
        """
        Check if the graph is directed.
        
        Returns:
            True if graph is directed, False otherwise.
        """
        return self._directed
        
    def is_weighted(self) -> bool:
        """
        Check if the graph is weighted.
        
        Returns:
            True if graph is weighted, False otherwise.
        """
        return self._weighted
        
    def degree(self, vertex: T) -> int:
        """
        Get the degree of a vertex.
        
        For directed graphs, this returns the out-degree.
        
        Args:
            vertex: The vertex to get degree for.
            
        Returns:
            The degree of the vertex.
            
        Raises:
            InvalidOperationError: If vertex does not exist.
        """
        if vertex not in self._adjacency_list:
            raise InvalidOperationError(f"Vertex {vertex} does not exist")
            
        return len(self._adjacency_list[vertex])
        
    def in_degree(self, vertex: T) -> int:
        """
        Get the in-degree of a vertex (for directed graphs).
        
        Args:
            vertex: The vertex to get in-degree for.
            
        Returns:
            The in-degree of the vertex.
            
        Raises:
            InvalidOperationError: If vertex does not exist or graph is undirected.
        """
        if not self._directed:
            raise InvalidOperationError("In-degree is only defined for directed graphs")
            
        if vertex not in self._adjacency_list:
            raise InvalidOperationError(f"Vertex {vertex} does not exist")
            
        in_deg = 0
        for v in self._adjacency_list:
            if vertex in self._adjacency_list[v]:
                in_deg += 1
        return in_deg
        
    def clear(self) -> None:
        """
        Remove all vertices and edges from the graph.
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._adjacency_list.clear()
        self._num_edges = 0
        
    def __len__(self) -> int:
        """Return the number of vertices in the graph."""
        return len(self._adjacency_list)
        
    def __contains__(self, vertex: T) -> bool:
        """Check if a vertex is in the graph."""
        return vertex in self._adjacency_list
        
    def __repr__(self) -> str:
        """Return detailed string representation."""
        direction = "directed" if self._directed else "undirected"
        weight_info = "weighted" if self._weighted else "unweighted"
        return f"Graph({direction}, {weight_info}, vertices={self.num_vertices()}, edges={self.num_edges()})"
        
    def __str__(self) -> str:
        """Return simple string representation."""
        vertices = list(self._adjacency_list.keys())
        if len(vertices) <= 5:
            return f"Graph{vertices}"
        else:
            return f"Graph[{len(vertices)} vertices, {self._num_edges} edges]"

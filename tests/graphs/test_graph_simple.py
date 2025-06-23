"""
Test suite for the Graph data structure.
"""

import pytest
from pyalgos.graphs.graph import Graph
from pyalgos.exceptions import InvalidOperationError


class TestGraphBasicOperations:
    """Test basic graph operations."""

    def test_empty_graph_creation(self) -> None:
        """Test creating an empty graph."""
        g = Graph[str](directed=False)
        assert len(g) == 0
        assert g.num_vertices() == 0
        assert g.num_edges() == 0
        assert not g.is_directed()

    def test_vertex_operations(self) -> None:
        """Test vertex operations."""
        g = Graph[str](directed=False)
        
        g.add_vertex("A")
        g.add_vertex("B")
        g.add_vertex("C")
        
        assert g.num_vertices() == 3
        assert g.has_vertex("A")
        assert "A" in g
        
        g.remove_vertex("B")
        assert g.num_vertices() == 2
        assert not g.has_vertex("B")

    def test_edge_operations(self) -> None:
        """Test edge operations."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        g.add_vertex("B")
        
        g.add_edge("A", "B")
        assert g.num_edges() == 1
        assert g.has_edge("A", "B")
        assert g.has_edge("B", "A")  # Undirected
        
        g.remove_edge("A", "B")
        assert g.num_edges() == 0
        assert not g.has_edge("A", "B")

    def test_weighted_operations(self) -> None:
        """Test weighted graph operations."""
        g = Graph[str](directed=False, weighted=True)
        assert g.is_weighted()
        
        g.add_vertex("A")
        g.add_vertex("B")
        g.add_edge("A", "B", weight=5.0)
        assert g.get_edge_weight("A", "B") == 5.0

    def test_neighbor_operations(self) -> None:
        """Test neighbor operations."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        g.add_vertex("B")
        g.add_vertex("C")
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        
        neighbors = g.get_neighbors("A")
        assert set(neighbors) == {"B", "C"}
        assert g.degree("A") == 2

    def test_directed_graph(self) -> None:
        """Test directed graph operations."""
        g = Graph[str](directed=True)
        assert g.is_directed()
        
        g.add_vertex("A")
        g.add_vertex("B")
        g.add_edge("A", "B")
        
        assert g.has_edge("A", "B")
        assert not g.has_edge("B", "A")
        assert g.in_degree("B") == 1
        assert g.in_degree("A") == 0

    def test_clear_operation(self) -> None:
        """Test clear operation."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        g.add_vertex("B")
        g.add_edge("A", "B")
        
        g.clear()
        assert g.num_vertices() == 0
        assert g.num_edges() == 0

    def test_get_vertices_and_edges(self) -> None:
        """Test getting vertices and edges."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        g.add_vertex("B")
        g.add_edge("A", "B")
        
        vertices = g.get_vertices()
        assert set(vertices) == {"A", "B"}
        
        edges = g.get_edges()
        assert len(edges) > 0

    def test_string_representation(self) -> None:
        """Test string representation."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        
        repr_str = repr(g)
        assert "Graph" in repr_str

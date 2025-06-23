"""
Test suite for graph traversal algorithms.
"""

import pytest
from pyalgos.graphs.graph import Graph
from pyalgos.graphs.traversal import BFS, DFS
from pyalgos.exceptions import InvalidOperationError


class TestBFS:
    """Test BFS traversal."""

    def test_bfs_traversal(self) -> None:
        """Test basic BFS traversal."""
        g = Graph[str](directed=False)
        vertices = ["A", "B", "C", "D"]
        for v in vertices:
            g.add_vertex(v)
        
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        
        bfs = BFS(g)
        result = bfs.traverse("A")
        assert "A" in result
        assert len(result) == 4

    def test_bfs_path_finding(self) -> None:
        """Test BFS path finding."""
        g = Graph[str](directed=False)
        vertices = ["A", "B", "C"]
        for v in vertices:
            g.add_vertex(v)
        
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        
        bfs = BFS(g)
        path = bfs.find_path("A", "C")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_bfs_distance(self) -> None:
        """Test BFS distance calculation."""
        g = Graph[str](directed=False)
        vertices = ["A", "B", "C"]
        for v in vertices:
            g.add_vertex(v)
        
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        
        bfs = BFS(g)
        distance = bfs.shortest_distance("A", "C")
        assert distance == 2


class TestDFS:
    """Test DFS traversal."""

    def test_dfs_traversal(self) -> None:
        """Test basic DFS traversal."""
        g = Graph[str](directed=False)
        vertices = ["A", "B", "C", "D"]
        for v in vertices:
            g.add_vertex(v)
        
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        
        dfs = DFS(g)
        result = dfs.traverse("A")
        assert "A" in result
        assert len(result) == 4

    def test_dfs_path_finding(self) -> None:
        """Test DFS path finding."""
        g = Graph[str](directed=False)
        vertices = ["A", "B", "C"]
        for v in vertices:
            g.add_vertex(v)
        
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        
        dfs = DFS(g)
        path = dfs.find_path("A", "C")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_cycle_detection(self) -> None:
        """Test cycle detection."""
        g = Graph[str](directed=False)
        vertices = ["A", "B", "C"]
        for v in vertices:
            g.add_vertex(v)
        
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        
        dfs = DFS(g)
        assert not dfs.has_cycle()
        
        # Add cycle
        g.add_edge("C", "A")
        assert dfs.has_cycle()


class TestBFSAdvanced:
    """Test advanced BFS functionality and error conditions."""

    def test_bfs_invalid_start_vertex(self) -> None:
        """Test BFS with invalid start vertex."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        
        bfs = BFS(g)
        with pytest.raises(InvalidOperationError):
            bfs.traverse("B")  # Non-existent vertex

    def test_bfs_path_invalid_vertices(self) -> None:
        """Test BFS path finding with invalid vertices."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        
        bfs = BFS(g)
        
        # Invalid start vertex
        with pytest.raises(InvalidOperationError):
            bfs.find_path("B", "A")
            
        # Invalid target vertex
        with pytest.raises(InvalidOperationError):
            bfs.find_path("A", "B")

    def test_bfs_same_start_target(self) -> None:
        """Test BFS path finding when start equals target."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        
        bfs = BFS(g)
        path = bfs.find_path("A", "A")
        assert path == ["A"]

    def test_bfs_no_path(self) -> None:
        """Test BFS when no path exists."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        g.add_vertex("B")  # Disconnected vertex
        
        bfs = BFS(g)
        path = bfs.find_path("A", "B")
        assert path is None

    def test_bfs_visitor_function(self) -> None:
        """Test BFS with visitor function."""
        g = Graph[str](directed=False)
        vertices = ["A", "B", "C"]
        for v in vertices:
            g.add_vertex(v)
        
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        
        visited_order = []
        def visitor(vertex: str) -> None:
            visited_order.append(vertex)
        
        bfs = BFS(g)
        bfs.traverse("A", visitor)
        assert "A" in visited_order
        assert len(visited_order) == 3

    def test_bfs_shortest_distance_invalid_vertices(self) -> None:
        """Test BFS shortest distance with invalid vertices."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        
        bfs = BFS(g)
        
        # Invalid start vertex
        with pytest.raises(InvalidOperationError):
            bfs.shortest_distance("B", "A")
            
        # Invalid target vertex
        with pytest.raises(InvalidOperationError):
            bfs.shortest_distance("A", "B")

    def test_bfs_shortest_distance_no_path(self) -> None:
        """Test BFS shortest distance when no path exists."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        g.add_vertex("B")  # Disconnected
        
        bfs = BFS(g)
        distance = bfs.shortest_distance("A", "B")
        assert distance is None

    def test_bfs_shortest_distance_same_vertex(self) -> None:
        """Test BFS shortest distance from vertex to itself."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        
        bfs = BFS(g)
        distance = bfs.shortest_distance("A", "A")
        assert distance == 0


class TestDFSAdvanced:
    """Test advanced DFS functionality and error conditions."""

    def test_dfs_invalid_start_vertex(self) -> None:
        """Test DFS with invalid start vertex."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        
        dfs = DFS(g)
        with pytest.raises(InvalidOperationError):
            dfs.traverse("B")  # Non-existent vertex

    def test_dfs_path_invalid_vertices(self) -> None:
        """Test DFS path finding with invalid vertices."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        
        dfs = DFS(g)
        
        # Invalid start vertex
        with pytest.raises(InvalidOperationError):
            dfs.find_path("B", "A")
            
        # Invalid target vertex
        with pytest.raises(InvalidOperationError):
            dfs.find_path("A", "B")

    def test_dfs_same_start_target(self) -> None:
        """Test DFS path finding when start equals target."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        
        dfs = DFS(g)
        path = dfs.find_path("A", "A")
        assert path == ["A"]

    def test_dfs_no_path(self) -> None:
        """Test DFS when no path exists."""
        g = Graph[str](directed=False)
        g.add_vertex("A")
        g.add_vertex("B")  # Disconnected vertex
        
        dfs = DFS(g)
        path = dfs.find_path("A", "B")
        assert path is None

    def test_dfs_visitor_function(self) -> None:
        """Test DFS with visitor function."""
        g = Graph[str](directed=False)
        vertices = ["A", "B", "C"]
        for v in vertices:
            g.add_vertex(v)
        
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        
        visited_order = []
        def visitor(vertex: str) -> None:
            visited_order.append(vertex)
        
        dfs = DFS(g)
        dfs.traverse("A", visitor)
        assert "A" in visited_order
        assert len(visited_order) == 3

    def test_dfs_iterative_vs_recursive(self) -> None:
        """Test that iterative DFS produces valid traversal."""
        g = Graph[str](directed=False)
        vertices = ["A", "B", "C", "D"]
        for v in vertices:
            g.add_vertex(v)
        
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        
        dfs = DFS(g)
        result = dfs.traverse("A")
        assert len(result) == 4
        assert "A" in result

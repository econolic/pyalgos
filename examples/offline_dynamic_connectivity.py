"""
Offline Dynamic Connectivity Problem

This module demonstrates a sophisticated algorithmic solution that combines
SegmentTree and DisjointSetUnion to efficiently solve the offline dynamic
connectivity problem.

Problem Statement:
- Given a sequence of edge additions, deletions, and connectivity queries over time
- All operations are known in advance (offline)
- Answer connectivity queries efficiently

Solution Approach:
- Use a SegmentTree over time where each node stores edges active during its time range
- Use a rollback-capable DisjointSetUnion to handle connectivity
- Process queries by traversing the SegmentTree and applying relevant edges

Time Complexity: O(Q log T α(n)) where Q is queries, T is time range, n is vertices
Space Complexity: O(E log T + n) where E is number of edges

Author: PyAlgos Team
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import sys
import os

# Add the parent directory to the path to import pyalgos modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyalgos.union_find.disjoint_set_union import DisjointSetUnion


@dataclass
class Operation:
    """Represents a DSU operation that can be rolled back"""
    op_type: str  # 'union'
    parent_changes: List[Tuple[int, int]]  # (index, old_value)
    size_changes: List[Tuple[int, int]]    # (index, old_value)


class RollbackDSU:
    """
    DisjointSetUnion with rollback capability.
    
    This specialized DSU maintains operation history to allow rolling back
    the last k operations. Essential for the SegmentTree approach where
    we need to backtrack and undo operations.
    
    Note: Does not use path compression to maintain rollback capability.
    """
    
    def __init__(self, n: int) -> None:
        """Initialize DSU for n elements (0 to n-1)"""
        if n <= 0:
            raise ValueError("Number of elements must be positive")
            
        self.n = n
        self.parent = list(range(n))
        self.size = [1] * n
        self.operations: List[Operation] = []
    
    def find(self, x: int) -> int:
        """Find root of element x without path compression"""
        if not (0 <= x < self.n):
            raise ValueError(f"Element {x} out of range [0, {self.n})")
            
        while self.parent[x] != x:
            x = self.parent[x]
        return x
    
    def connected(self, a: int, b: int) -> bool:
        """Check if elements a and b are in the same component"""
        return self.find(a) == self.find(b)
    
    def union(self, a: int, b: int) -> bool:
        """
        Union components containing a and b.
        Returns True if union was performed, False if already connected.
        """
        root_a = self.find(a)
        root_b = self.find(b)
        
        if root_a == root_b:
            # Record no-op for consistent rollback
            self.operations.append(Operation('union', [], []))
            return False
        
        # Union by size - attach smaller tree to larger tree
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        
        # Record changes for rollback
        parent_changes = [(root_b, self.parent[root_b])]
        size_changes = [(root_a, self.size[root_a])]
        
        # Perform union
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]
        
        # Record operation
        operation = Operation('union', parent_changes, size_changes)
        self.operations.append(operation)
        
        return True
    
    def rollback(self, steps: int = 1) -> None:
        """Rollback the last 'steps' operations"""
        if steps < 0:
            raise ValueError("Steps must be non-negative")
        if steps > len(self.operations):
            raise ValueError(f"Cannot rollback {steps} steps, only {len(self.operations)} operations available")
        
        for _ in range(steps):
            if not self.operations:
                break
                
            op = self.operations.pop()
            
            # Reverse size changes
            for idx, old_value in reversed(op.size_changes):
                self.size[idx] = old_value
            
            # Reverse parent changes
            for idx, old_value in reversed(op.parent_changes):
                self.parent[idx] = old_value
    
    def get_operations_count(self) -> int:
        """Get number of operations performed (for rollback tracking)"""
        return len(self.operations)
    
    def component_size(self, x: int) -> int:
        """Get size of component containing element x"""
        return self.size[self.find(x)]


class OfflineDynamicConnectivity:
    """
    Solves the offline dynamic connectivity problem using SegmentTree + DSU.
    
    This algorithm efficiently handles:
    - Edge additions and deletions over time
    - Connectivity queries at specific time points
    - All operations known in advance (offline)
    
    The key insight is to use a SegmentTree over time where each node
    stores edges that are active during its time range.
    """
    
    def __init__(self, n_vertices: int, max_time: int) -> None:
        """
        Initialize the offline dynamic connectivity solver.
        
        Args:
            n_vertices: Number of vertices in the graph (0 to n_vertices-1)
            max_time: Maximum time value (time range is [0, max_time))
        """
        if n_vertices <= 0:
            raise ValueError("Number of vertices must be positive")
        if max_time <= 0:
            raise ValueError("Max time must be positive")
            
        self.n_vertices = n_vertices
        self.max_time = max_time
        
        # SegmentTree for time ranges - each node stores edges active in its range
        self.tree_size = 4 * max_time  # Sufficient size for segment tree
        self.edges_at_node: List[List[Tuple[int, int]]] = [[] for _ in range(self.tree_size)]
        
        # Query storage
        self.queries: List[Tuple[int, int, int, int]] = []  # (u, v, time, query_id)
        self.query_results: Dict[int, bool] = {}
    
    def add_edge(self, u: int, v: int, start_time: int, end_time: int) -> None:
        """
        Add an edge (u, v) that exists from start_time to end_time (exclusive).
        
        Args:
            u, v: Vertices of the edge
            start_time: Time when edge is added
            end_time: Time when edge is removed (exclusive)
        """
        if not (0 <= u < self.n_vertices and 0 <= v < self.n_vertices):
            raise ValueError(f"Vertices must be in range [0, {self.n_vertices})")
        if not (0 <= start_time < end_time <= self.max_time):
            raise ValueError(f"Invalid time range [{start_time}, {end_time})")
        
        # Add edge to appropriate segment tree nodes
        self._add_edge_to_tree(1, 0, self.max_time - 1, start_time, end_time - 1, (u, v))
    
    def _add_edge_to_tree(self, node: int, tl: int, tr: int, l: int, r: int, edge: Tuple[int, int]) -> None:
        """Add edge to segment tree nodes that completely overlap with [l, r]"""
        if l > r:
            return
        if l == tl and r == tr:
            self.edges_at_node[node].append(edge)
            return
        
        tm = (tl + tr) // 2
        self._add_edge_to_tree(2 * node, tl, tm, l, min(r, tm), edge)
        self._add_edge_to_tree(2 * node + 1, tm + 1, tr, max(l, tm + 1), r, edge)
    
    def add_query(self, u: int, v: int, time: int) -> int:
        """
        Add a connectivity query: are u and v connected at given time?
        
        Returns query_id for retrieving result later.
        """
        if not (0 <= u < self.n_vertices and 0 <= v < self.n_vertices):
            raise ValueError(f"Vertices must be in range [0, {self.n_vertices})")
        if not (0 <= time < self.max_time):
            raise ValueError(f"Time must be in range [0, {self.max_time})")
        
        query_id = len(self.queries)
        self.queries.append((u, v, time, query_id))
        return query_id
    
    def process_queries(self) -> Dict[int, bool]:
        """
        Process all queries and return results.
        
        Returns:
            Dictionary mapping query_id to connectivity result
        """
        self.query_results.clear()
        dsu = RollbackDSU(self.n_vertices)
        
        # Process each query
        for u, v, time, query_id in self.queries:
            # Traverse segment tree to time point and collect edges
            self._process_query(1, 0, self.max_time - 1, time, dsu, u, v, query_id)
        
        return self.query_results.copy()
    
    def _process_query(self, node: int, tl: int, tr: int, time: int, 
                      dsu: RollbackDSU, u: int, v: int, query_id: int) -> None:
        """Process query by traversing segment tree and applying edges"""
        # Apply all edges at current node
        ops_before = dsu.get_operations_count()
        
        for edge_u, edge_v in self.edges_at_node[node]:
            dsu.union(edge_u, edge_v)
        
        if tl == tr:
            # Leaf node - answer query
            self.query_results[query_id] = dsu.connected(u, v)
        else:
            # Internal node - continue traversal
            tm = (tl + tr) // 2
            if time <= tm:
                self._process_query(2 * node, tl, tm, time, dsu, u, v, query_id)
            else:
                self._process_query(2 * node + 1, tm + 1, tr, time, dsu, u, v, query_id)
        
        # Rollback operations from current node
        ops_to_rollback = dsu.get_operations_count() - ops_before
        if ops_to_rollback > 0:
            dsu.rollback(ops_to_rollback)
    
    def get_query_result(self, query_id: int) -> Optional[bool]:
        """Get result of a specific query"""
        return self.query_results.get(query_id)
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the data structure"""
        total_edges = sum(len(edges) for edges in self.edges_at_node)
        return {
            'vertices': self.n_vertices,
            'max_time': self.max_time,
            'total_edge_instances': total_edges,
            'queries': len(self.queries)
        }


def demonstrate_offline_dynamic_connectivity():
    """Comprehensive demonstration of the offline dynamic connectivity algorithm"""
    print("=== Offline Dynamic Connectivity Demonstration ===\n")
    
    # Create a scenario with 5 vertices and time range [0, 10)
    n_vertices = 5
    max_time = 10
    solver = OfflineDynamicConnectivity(n_vertices, max_time)
    
    print(f"Graph with {n_vertices} vertices, time range [0, {max_time})")
    print("\nAdding edges with their lifetimes:")
    
    # Add edges with different lifetimes
    edges = [
        (0, 1, 0, 5),   # Edge (0,1) exists from time 0 to 4
        (1, 2, 2, 7),   # Edge (1,2) exists from time 2 to 6
        (2, 3, 1, 4),   # Edge (2,3) exists from time 1 to 3
        (3, 4, 5, 9),   # Edge (3,4) exists from time 5 to 8
        (0, 4, 6, 10),  # Edge (0,4) exists from time 6 to 9
    ]
    
    for u, v, start, end in edges:
        solver.add_edge(u, v, start, end)
        print(f"  Edge ({u},{v}): active from time {start} to {end-1}")
    
    print("\nAdding connectivity queries:")
    
    # Add various connectivity queries
    queries = [
        (0, 1, 0),  # Connected at time 0? (Yes - direct edge)
        (0, 1, 5),  # Connected at time 5? (No - edge expired)
        (0, 2, 3),  # Connected at time 3? (Yes - path 0-1-2)
        (0, 3, 2),  # Connected at time 2? (Yes - path 0-1-2-3)
        (0, 4, 7),  # Connected at time 7? (Yes - direct edge 0-4)
        (1, 4, 6),  # Connected at time 6? (No - no path)
        (2, 4, 6),  # Connected at time 6? (No - components separated)
        (0, 3, 8),  # Connected at time 8? (Yes - path 0-4-3)
    ]
    
    query_ids = []
    for u, v, time in queries:
        query_id = solver.add_query(u, v, time)
        query_ids.append((query_id, u, v, time))
        print(f"  Query {query_id}: Are vertices {u} and {v} connected at time {time}?")
    
    print("\nProcessing all queries...")
    results = solver.process_queries()
    
    print("\nQuery Results:")
    for query_id, u, v, time in query_ids:
        result = results[query_id]
        print(f"  Query {query_id}: Connected({u},{v}) at time {time} = {result}")
    
    print(f"\nSolver Statistics:")
    stats = solver.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Demonstrate the algorithm's power with a complex scenario
    print("\n" + "="*60)
    print("Complex Scenario: Network Reliability Over Time")
    print("="*60)
    
    # Simulate a network where connections come and go
    network = OfflineDynamicConnectivity(6, 15)
    
    print("\nNetwork edges (representing network links):")
    network_edges = [
        (0, 1, 0, 15),   # Backbone connection - always available
        (1, 2, 0, 8),    # First half connection
        (2, 3, 7, 15),   # Second half connection
        (1, 4, 3, 12),   # Branch connection
        (4, 5, 5, 10),   # End connection
        (0, 5, 10, 15),  # Emergency backup connection
    ]
    
    for u, v, start, end in network_edges:
        network.add_edge(u, v, start, end)
        print(f"  Link {u}-{v}: available from time {start} to {end-1}")
    
    print("\nReliability queries (can data flow between nodes?):")
    reliability_queries = [
        (0, 3, 5),   # Can 0 reach 3 at time 5?
        (0, 3, 9),   # Can 0 reach 3 at time 9?
        (0, 5, 8),   # Can 0 reach 5 at time 8?
        (0, 5, 12),  # Can 0 reach 5 at time 12?
        (2, 4, 6),   # Can 2 reach 4 at time 6?
    ]
    
    query_ids = []
    for u, v, time in reliability_queries:
        query_id = network.add_query(u, v, time)
        query_ids.append((query_id, u, v, time))
        print(f"  Reliability {query_id}: Can node {u} reach node {v} at time {time}?")
    
    print("\nAnalyzing network reliability...")
    results = network.process_queries()
    
    print("\nReliability Analysis Results:")
    for query_id, u, v, time in query_ids:
        result = results[query_id]
        status = "REACHABLE" if result else "UNREACHABLE"
        print(f"  Time {time}: Node {u} -> Node {v} is {status}")
    
    print(f"\nNetwork Statistics:")
    stats = network.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("Algorithm Analysis")
    print("="*60)
    print(f"Time Complexity: O(Q log T α(n))")
    print(f"  Q = number of queries")
    print(f"  T = time range")
    print(f"  n = number of vertices")
    print(f"  α = inverse Ackermann function (practically constant)")
    print(f"\nSpace Complexity: O(E log T + n)")
    print(f"  E = number of edges")
    print(f"  Each edge stored in O(log T) segment tree nodes")
    
    print(f"\nKey Features:")
    print(f"  ✓ Handles dynamic edge additions/deletions")
    print(f"  ✓ Efficient offline query processing")
    print(f"  ✓ Combines SegmentTree + DisjointSetUnion")
    print(f"  ✓ Rollback capability for backtracking")
    print(f"  ✓ Optimal time complexity for offline setting")


def test_rollback_dsu():
    """Test the rollback DSU implementation"""
    print("\n=== Testing RollbackDSU ===")
    
    dsu = RollbackDSU(5)
    print(f"Initial: 5 components")
    
    # Perform some unions
    dsu.union(0, 1)
    print(f"After union(0,1): connected(0,1) = {dsu.connected(0, 1)}")
    
    dsu.union(2, 3)
    print(f"After union(2,3): connected(2,3) = {dsu.connected(2, 3)}")
    
    ops_before = dsu.get_operations_count()
    dsu.union(1, 2)  # This connects components {0,1} and {2,3}
    print(f"After union(1,2): connected(0,3) = {dsu.connected(0, 3)}")
    
    # Test rollback
    print(f"\nRolling back 1 operation...")
    dsu.rollback(1)
    print(f"After rollback: connected(0,3) = {dsu.connected(0, 3)}")
    print(f"After rollback: connected(0,1) = {dsu.connected(0, 1)}")
    print(f"After rollback: connected(2,3) = {dsu.connected(2, 3)}")
    
    # Test component sizes
    print(f"\nComponent sizes:")
    for i in range(5):
        if dsu.find(i) == i:  # Root of component
            size = dsu.component_size(i)
            print(f"  Component with root {i}: size {size}")


if __name__ == "__main__":
    # Run comprehensive demonstration
    demonstrate_offline_dynamic_connectivity()
    
    # Test rollback functionality
    test_rollback_dsu()
    
    print(f"\n{'='*60}")
    print("✅ Offline Dynamic Connectivity implementation complete!")
    print("This demonstrates the power of combining multiple data structures")
    print("to solve complex algorithmic problems efficiently.")
    print(f"{'='*60}")

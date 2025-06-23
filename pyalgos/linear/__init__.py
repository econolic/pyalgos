"""Linear data structures package.

This package provides implementations of fundamental linear data structures
including stacks, queues, and linked lists. All implementations follow
modern Python best practices with comprehensive type hints and documentation.
"""

from __future__ import annotations

# Import order matters for proper initialization
try:
    from pyalgos.linear.stack import Stack
    from pyalgos.linear.queue import Queue
    from pyalgos.linear.linked_list import LinkedList
except ImportError:
    # Fallback for development/testing
    from .stack import Stack
    from .queue import Queue
    from .linked_list import LinkedList

__all__ = ["Stack", "Queue", "LinkedList"]

"""
Stack Usage Examples

This module demonstrates practical applications of the Stack data structure
in various algorithmic and real-world scenarios.

Examples include:
- Balanced parentheses checking
- Expression evaluation (postfix)
- Function call simulation
- Undo/Redo functionality
- Backtracking algorithms

Author: PyAlgos Team
"""

import sys
import os
from typing import List, Union, Optional

# Add the parent directory to the path to import pyalgos modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyalgos.linear.stack import Stack


def check_balanced_parentheses(expression: str) -> bool:
    """
    Check if parentheses are balanced in an expression.
    
    Supports: (), [], {}
    
    Args:
        expression: String containing parentheses to check
        
    Returns:
        True if balanced, False otherwise
        
    Example:
        >>> check_balanced_parentheses("(a + b) * [c - d]")
        True
        >>> check_balanced_parentheses("((a + b)")
        False
    """
    stack = Stack[str]()
    opening = {'(', '[', '{'}
    closing = {')', ']', '}'}
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in expression:
        if char in opening:
            stack.push(char)
        elif char in closing:
            if stack.is_empty():
                return False
            
            top = stack.pop()
            if pairs[top] != char:
                return False
    
    return stack.is_empty()


def evaluate_postfix(expression: str) -> float:
    """
    Evaluate a postfix (Reverse Polish Notation) expression.
    
    Args:
        expression: Space-separated postfix expression
        
    Returns:
        Result of the expression
        
    Example:
        >>> evaluate_postfix("3 4 + 2 * 7 /")
        2.0  # ((3 + 4) * 2) / 7
    """
    stack = Stack[float]()
    tokens = expression.split()
    
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            if len(stack) < 2:
                raise ValueError("Invalid postfix expression")
                  b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                if b == 0:
                    raise ValueError("Division by zero")
                result = a / b
            else:
                raise ValueError(f"Unknown operator: {token}")
            
            stack.push(result)
        else:
            try:
                stack.push(float(token))
            except ValueError:
                raise ValueError(f"Invalid number: {token}")
    
    if len(stack) != 1:
        raise ValueError("Invalid postfix expression")
    
    return stack.pop()


class UndoRedoSystem:
    """
    Undo/Redo functionality using two stacks.
    
    This demonstrates how stacks can be used to implement
    undo/redo functionality in applications.
    """
    
    def __init__(self):
        self.undo_stack = Stack[str]()
        self.redo_stack = Stack[str]()
        self.current_state = ""
    
    def execute_command(self, command: str) -> None:
        """Execute a command and save current state for undo"""
        self.undo_stack.push(self.current_state)
        self.current_state = command
        # Clear redo stack when new command is executed
        self.redo_stack.clear()
        print(f"Executed: {command}")
    
    def undo(self) -> bool:
        """Undo the last command"""
        if self.undo_stack.is_empty():
            print("Nothing to undo")
            return False
        
        # Save current state for redo
        self.redo_stack.push(self.current_state)
        
        # Restore previous state
        self.current_state = self.undo_stack.pop()
        print(f"Undone. Current state: {self.current_state}")
        return True
    
    def redo(self) -> bool:
        """Redo the last undone command"""
        if self.redo_stack.is_empty():
            print("Nothing to redo")
            return False
        
        # Save current state for undo
        self.undo_stack.push(self.current_state)
        
        # Restore redone state
        self.current_state = self.redo_stack.pop()
        print(f"Redone. Current state: {self.current_state}")
        return True
    
    def get_current_state(self) -> str:
        """Get current state"""
        return self.current_state


def solve_tower_of_hanoi(n: int, source: str = "A", destination: str = "C", auxiliary: str = "B") -> List[str]:
    """
    Solve Tower of Hanoi puzzle using stack simulation.
    
    Args:
        n: Number of disks
        source: Source rod name
        destination: Destination rod name
        auxiliary: Auxiliary rod name
        
    Returns:
        List of moves to solve the puzzle
    """
    moves = []
    
    def hanoi_recursive(n: int, src: str, dest: str, aux: str):
        if n == 1:
            moves.append(f"Move disk 1 from {src} to {dest}")
        else:
            # Move n-1 disks from source to auxiliary
            hanoi_recursive(n-1, src, aux, dest)
            
            # Move the largest disk from source to destination
            moves.append(f"Move disk {n} from {src} to {dest}")
            
            # Move n-1 disks from auxiliary to destination
            hanoi_recursive(n-1, aux, dest, src)
    
    hanoi_recursive(n, source, destination, auxiliary)
    return moves


class FunctionCallSimulator:
    """
    Simulate function call stack to demonstrate recursion.
    
    This shows how the call stack works internally when
    functions call each other.
    """
    
    def __init__(self):
        self.call_stack = Stack[str]()
        self.call_depth = 0
    
    def call_function(self, function_name: str, *args) -> None:
        """Simulate calling a function"""
        call_info = f"{function_name}({', '.join(map(str, args))})"
        self.call_stack.push(call_info)
        self.call_depth += 1
        
        print("  " * (self.call_depth - 1) + f"→ Calling {call_info}")
    
    def return_from_function(self, return_value: Optional[str] = None) -> None:
        """Simulate returning from a function"""
        if self.call_stack.is_empty():
            print("No function to return from!")
            return
        
        returned_function = self.call_stack.pop()
        self.call_depth -= 1
        
        return_msg = f"← Returned from {returned_function}"
        if return_value is not None:
            return_msg += f" with value: {return_value}"
        
        print("  " * self.call_depth + return_msg)
    
    def simulate_factorial(self, n: int) -> int:
        """Simulate factorial calculation with call stack visualization"""
        print(f"\nSimulating factorial({n}):")
        
        def factorial_helper(num: int) -> int:
            self.call_function("factorial", num)
            
            if num <= 1:
                result = 1
                self.return_from_function(str(result))
                return result
            else:
                result = num * factorial_helper(num - 1)
                self.return_from_function(str(result))
                return result
        
        return factorial_helper(n)


def find_next_greater_elements(arr: List[int]) -> List[int]:
    """
    Find the next greater element for each element in the array.
    
    Uses stack to efficiently solve in O(n) time.
    
    Args:
        arr: List of integers
        
    Returns:
        List where result[i] is the next greater element than arr[i],
        or -1 if no such element exists
        
    Example:
        >>> find_next_greater_elements([4, 5, 2, 25])
        [5, 25, 25, -1]
    """
    result = [-1] * len(arr)
    stack = Stack[int]()  # Store indices
    
    for i in range(len(arr)):
        # Pop elements from stack while stack is not empty and
        # current element is greater than stack top element
        while not stack.is_empty() and arr[i] > arr[stack.peek()]:
            index = stack.pop()
            result[index] = arr[i]
        
        stack.push(i)
    
    return result


def validate_html_tags(html: str) -> bool:
    """
    Validate HTML tag matching using stack.
    
    Args:
        html: HTML string to validate
        
    Returns:
        True if tags are properly nested, False otherwise
    """
    stack = Stack[str]()
    i = 0
    
    while i < len(html):
        if html[i] == '<':
            # Find the end of the tag
            j = i + 1
            while j < len(html) and html[j] != '>':
                j += 1
            
            if j == len(html):
                return False  # Unclosed tag
            
            tag = html[i+1:j]
            
            if tag.startswith('/'):
                # Closing tag
                tag_name = tag[1:]
                if stack.is_empty():
                    return False  # Closing tag without opening
                
                opening_tag = stack.pop()
                if opening_tag != tag_name:
                    return False  # Mismatched tags
            else:
                # Opening tag (ignore self-closing tags like <br/>)
                if not tag.endswith('/'):
                    # Remove attributes for comparison
                    tag_name = tag.split()[0]
                    stack.push(tag_name)
            
            i = j + 1
        else:
            i += 1
    
    return stack.is_empty()


def main():
    """Demonstrate all stack usage examples"""
    print("Stack Usage Examples")
    print("=" * 50)
    
    # 1. Balanced Parentheses
    print("\n1. Balanced Parentheses Checking:")
    expressions = [
        "(a + b) * [c - d]",
        "((a + b)",
        "{[()]}",
        "([)]"
    ]
    
    for expr in expressions:
        is_balanced = check_balanced_parentheses(expr)
        print(f"  '{expr}' -> {'Balanced' if is_balanced else 'Not Balanced'}")
    
    # 2. Postfix Evaluation
    print("\n2. Postfix Expression Evaluation:")
    postfix_expressions = [
        "3 4 + 2 * 7 /",  # ((3 + 4) * 2) / 7 = 2
        "15 7 1 1 + - / 3 * 2 1 1 + + -",  # Complex expression
        "5 1 2 + 4 * + 3 -"  # 5 + ((1 + 2) * 4) - 3 = 14
    ]
    
    for expr in postfix_expressions:
        try:
            result = evaluate_postfix(expr)
            print(f"  '{expr}' = {result}")
        except ValueError as e:
            print(f"  '{expr}' -> Error: {e}")
    
    # 3. Undo/Redo System
    print("\n3. Undo/Redo System:")
    undo_redo = UndoRedoSystem()
    
    undo_redo.execute_command("Type 'Hello'")
    undo_redo.execute_command("Type ' World'")
    undo_redo.execute_command("Delete 'World'")
    
    print("  Current state:", undo_redo.get_current_state())
    
    undo_redo.undo()
    undo_redo.undo()
    undo_redo.redo()
    
    # 4. Tower of Hanoi
    print("\n4. Tower of Hanoi (n=3):")
    moves = solve_tower_of_hanoi(3)
    for i, move in enumerate(moves, 1):
        print(f"  Step {i}: {move}")
    
    # 5. Function Call Simulation
    print("\n5. Function Call Stack Simulation:")
    simulator = FunctionCallSimulator()
    result = simulator.simulate_factorial(4)
    print(f"  Final result: {result}")
    
    # 6. Next Greater Elements
    print("\n6. Next Greater Elements:")
    arrays = [
        [4, 5, 2, 25],
        [13, 7, 6, 12],
        [1, 2, 3, 4, 5]
    ]
    
    for arr in arrays:
        nge = find_next_greater_elements(arr)
        print(f"  {arr} -> {nge}")
    
    # 7. HTML Tag Validation
    print("\n7. HTML Tag Validation:")
    html_samples = [
        "<div><p>Hello</p></div>",
        "<div><p>Hello</div></p>",
        "<html><body><h1>Title</h1></body></html>",
        "<div><p>Unclosed tag"
    ]
    
    for html in html_samples:
        is_valid = validate_html_tags(html)
        print(f"  '{html}' -> {'Valid' if is_valid else 'Invalid'}")
    
    print("\n" + "=" * 50)
    print("Stack examples completed!")


if __name__ == "__main__":
    main()

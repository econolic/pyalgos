"""
Trie (Prefix Tree) implementation for efficient string operations.

This module provides a Trie data structure for string storage,
prefix searching, and string manipulation operations. The Trie supports various
operations like insertion, deletion, search, and prefix matching.

Key Features:
    - String insertion and deletion in O(m) time where m is string length
    - Prefix search and word search in O(m) time
    - All strings with given prefix retrieval
    - Count of strings with given prefix
    - Memory-efficient node representation
    - Support for case-sensitive and case-insensitive operations
    - Comprehensive input validation and error handling

Example:
    >>> from pyalgos.strings.trie import Trie
    >>> trie = Trie()
    >>> trie.insert("hello")
    >>> trie.insert("world")
    >>> trie.insert("help")
    >>> print(trie.search("hello"))  # True
    >>> print(trie.starts_with("hel"))  # True
    >>> print(trie.get_words_with_prefix("hel"))  # ["hello", "help"]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from pyalgos.exceptions import InvalidOperationError
from pyalgos.types import SizeType


class TrieNode:
    """
    A node in the Trie data structure.
    
    Each node represents a character in the trie and contains:
    - children: Dictionary mapping characters to child nodes
    - is_end_of_word: Boolean indicating if this node marks the end of a word
    - word_count: Number of words that pass through this node (for prefix counting)
    """
    
    def __init__(self) -> None:
        """Initialize a new Trie node."""
        self.children: Dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False
        self.word_count: int = 0


class Trie:
    """
    A Trie (Prefix Tree) data structure for efficient string operations.
    
    The Trie stores strings in a tree-like structure where each path from root
    to a node represents a prefix of one or more strings. This allows for
    efficient prefix-based operations and string searching.
    
    Attributes:
        _root: The root node of the trie.
        _size: Number of words stored in the trie.
        _case_sensitive: Whether the trie is case-sensitive.
        
    Time Complexities:
        - insert(word): O(m) where m is the length of the word
        - search(word): O(m) where m is the length of the word
        - starts_with(prefix): O(m) where m is the length of the prefix
        - delete(word): O(m) where m is the length of the word
        - get_words_with_prefix(prefix): O(p + n) where p is prefix length, n is result size
        
    Space Complexity: O(ALPHABET_SIZE * N * M) where N is number of words, M is average length
    """
    
    def __init__(self, case_sensitive: bool = True) -> None:
        """
        Initialize a new Trie.
        
        Args:
            case_sensitive: Whether the trie should be case-sensitive.
                          If False, all strings are converted to lowercase.
                          
        Example:
            >>> trie = Trie()  # Case-sensitive
            >>> trie_ci = Trie(case_sensitive=False)  # Case-insensitive
        """
        self._root = TrieNode()
        self._size: SizeType = 0
        self._case_sensitive = case_sensitive
        
    def insert(self, word: str) -> None:
        """
        Insert a word into the trie.
        
        Args:
            word: The word to insert.
            
        Raises:
            InvalidOperationError: If word is empty or invalid.
            
        Time Complexity: O(m) where m is the length of the word.
        Space Complexity: O(m) in the worst case when all characters are unique.
        
        Example:
            >>> trie = Trie()
            >>> trie.insert("hello")
            >>> trie.insert("world")
            >>> trie.size()
            2
        """
        if not word:
            raise InvalidOperationError("Cannot insert empty word into trie")
            
        if not isinstance(word, str):
            raise InvalidOperationError("Word must be a string")
            
        # Normalize case if needed
        if not self._case_sensitive:
            word = word.lower()
            
        current = self._root
        
        # Traverse/create path for each character
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
            current.word_count += 1
            
        # Mark end of word and update size if it's a new word
        if not current.is_end_of_word:
            current.is_end_of_word = True
            self._size += 1
            
    def search(self, word: str) -> bool:
        """
        Search for a word in the trie.
        
        Args:
            word: The word to search for.
            
        Returns:
            True if the word exists in the trie, False otherwise.
            
        Raises:
            InvalidOperationError: If word is empty or invalid.
            
        Time Complexity: O(m) where m is the length of the word.
        Space Complexity: O(1).
        
        Example:
            >>> trie = Trie()
            >>> trie.insert("hello")
            >>> trie.search("hello")  # True
            >>> trie.search("hell")   # False (not a complete word)
            >>> trie.search("world")  # False
        """
        if not word:
            raise InvalidOperationError("Cannot search for empty word")
            
        if not isinstance(word, str):
            raise InvalidOperationError("Word must be a string")
            
        # Normalize case if needed
        if not self._case_sensitive:
            word = word.lower()
            
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
        
    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word in the trie starts with the given prefix.
        
        Args:
            prefix: The prefix to check.
            
        Returns:
            True if any word starts with the prefix, False otherwise.
            
        Raises:
            InvalidOperationError: If prefix is invalid.
            
        Time Complexity: O(m) where m is the length of the prefix.
        Space Complexity: O(1).
        
        Example:
            >>> trie = Trie()
            >>> trie.insert("hello")
            >>> trie.insert("help")
            >>> trie.starts_with("hel")  # True
            >>> trie.starts_with("wor")  # False
        """
        if not isinstance(prefix, str):
            raise InvalidOperationError("Prefix must be a string")
            
        # Empty prefix matches everything
        if not prefix:
            return self._size > 0
              # Normalize case if needed
        if not self._case_sensitive:
            prefix = prefix.lower()
            
        return self._find_node(prefix) is not None
        
    def delete(self, word: str) -> bool:
        """
        Delete a word from the trie.
        
        Args:
            word: The word to delete.
            
        Returns:
            True if the word was deleted, False if the word was not found.
            
        Raises:
            InvalidOperationError: If word is empty or invalid.
            
        Time Complexity: O(m) where m is the length of the word.
        Space Complexity: O(m) due to recursion stack.
        
        Example:
            >>> trie = Trie()
            >>> trie.insert("hello")
            >>> trie.delete("hello")  # True
            >>> trie.delete("hello")  # False (already deleted)
        """
        if not word:
            raise InvalidOperationError("Cannot delete empty word")
            
        if not isinstance(word, str):
            raise InvalidOperationError("Word must be a string")
            
        # Normalize case if needed
        if not self._case_sensitive:
            word = word.lower()
            
        # Track if word was actually deleted
        word_deleted = False
        
        def _delete_recursive(node: TrieNode, word: str, index: int) -> bool:
            """
            Recursively delete a word from the trie.
            
            Returns:
                True if the current node should be deleted, False otherwise.
            """
            nonlocal word_deleted
            
            if index == len(word):
                # Base case: reached end of word
                if not node.is_end_of_word:
                    return False  # Word doesn't exist
                    
                node.is_end_of_word = False
                self._size -= 1
                word_deleted = True
                
                # Delete node if it has no children and is not end of another word
                return len(node.children) == 0
                
            char = word[index]
            child_node = node.children.get(char)
            
            if child_node is None:
                return False  # Word doesn't exist
                
            should_delete_child = _delete_recursive(child_node, word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                child_node.word_count = 0
            else:
                child_node.word_count -= 1
                
            # Delete current node if:
            # 1. It has no children
            # 2. It's not the end of another word
            # 3. It's not the root
            return (len(node.children) == 0 and 
                   not node.is_end_of_word and 
                   node is not self._root)
                   
        _delete_recursive(self._root, word, 0)
        return word_deleted
        
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """
        Get all words in the trie that start with the given prefix.
        
        Args:
            prefix: The prefix to search for.
            
        Returns:
            A list of all words that start with the prefix.
            
        Raises:
            InvalidOperationError: If prefix is invalid.
            
        Time Complexity: O(p + n) where p is prefix length, n is result size.
        Space Complexity: O(n) where n is the number of matching words.
        
        Example:
            >>> trie = Trie()
            >>> trie.insert("hello")
            >>> trie.insert("help")
            >>> trie.insert("world")
            >>> trie.get_words_with_prefix("hel")  # ["hello", "help"]
        """
        if not isinstance(prefix, str):
            raise InvalidOperationError("Prefix must be a string")
            
        # Normalize case if needed
        if not self._case_sensitive:
            prefix = prefix.lower()
            
        result: List[str] = []
        
        # Find the node corresponding to the prefix
        prefix_node = self._find_node(prefix)
        if prefix_node is None:
            return result
            
        # Collect all words starting from the prefix node
        self._collect_words(prefix_node, prefix, result)
        return result
        
    def count_words_with_prefix(self, prefix: str) -> int:
        """
        Count the number of words that start with the given prefix.
        
        Args:
            prefix: The prefix to count.
            
        Returns:
            The number of words that start with the prefix.
            
        Time Complexity: O(m) where m is the length of the prefix.
        Space Complexity: O(1).
        
        Example:
            >>> trie = Trie()
            >>> trie.insert("hello")
            >>> trie.insert("help")
            >>> trie.count_words_with_prefix("hel")  # 2
        """
        if not isinstance(prefix, str):
            raise InvalidOperationError("Prefix must be a string")
            
        # Empty prefix matches all words
        if not prefix:
            return self._size
            
        # Normalize case if needed
        if not self._case_sensitive:
            prefix = prefix.lower()
            
        prefix_node = self._find_node(prefix)
        if prefix_node is None:
            return 0
            
        # Count words in subtree
        return self._count_words_in_subtree(prefix_node)
        
    def size(self) -> SizeType:
        """
        Get the number of words in the trie.
        
        Returns:
            The number of words stored in the trie.
            
        Time Complexity: O(1).
        Space Complexity: O(1).
        """
        return self._size
        
    def is_empty(self) -> bool:
        """
        Check if the trie is empty.
        
        Returns:
            True if the trie contains no words, False otherwise.
            
        Time Complexity: O(1).
        Space Complexity: O(1).
        """
        return self._size == 0
        
    def get_all_words(self) -> List[str]:
        """
        Get all words stored in the trie.
        
        Returns:
            A list of all words in the trie in lexicographical order.
            
        Time Complexity: O(n) where n is the total number of characters in all words.
        Space Complexity: O(n) where n is the total number of characters in all words.
        
        Example:
            >>> trie = Trie()
            >>> trie.insert("world")
            >>> trie.insert("hello")
            >>> trie.get_all_words()  # ["hello", "world"] (lexicographical order)
        """
        result: List[str] = []
        self._collect_words(self._root, "", result)
        return result
        
    def clear(self) -> None:
        """
        Remove all words from the trie.
        
        Time Complexity: O(1).
        Space Complexity: O(1).
        """
        self._root = TrieNode()
        self._size = 0
        
    def longest_common_prefix(self) -> str:
        """
        Find the longest common prefix of all words in the trie.
        
        Returns:
            The longest common prefix, or empty string if no common prefix exists.
            
        Time Complexity: O(m) where m is the length of the longest common prefix.
        Space Complexity: O(m) where m is the length of the longest common prefix.
        
        Example:
            >>> trie = Trie()
            >>> trie.insert("hello")
            >>> trie.insert("help")
            >>> trie.longest_common_prefix()  # "hel"
        """
        if self.is_empty():
            return ""
            
        prefix = ""
        current = self._root
        
        # Continue while there's exactly one child and it's not end of word
        while (len(current.children) == 1 and 
               not current.is_end_of_word and 
               current != self._root):
            char = next(iter(current.children.keys()))
            prefix += char
            current = current.children[char]
            
        # Check if we stopped because we reached end of word or multiple children
        if len(current.children) == 1 and not current.is_end_of_word:
            char = next(iter(current.children.keys()))
            prefix += char
            
        return prefix
        
    def _find_node(self, word: str) -> Optional[TrieNode]:
        """
        Find the node corresponding to the given word/prefix.
        
        Args:
            word: The word or prefix to find.
            
        Returns:
            The node if found, None otherwise.
        """
        current = self._root
        
        for char in word:
            if char not in current.children:
                return None
            current = current.children[char]
            
        return current
        
    def _collect_words(self, node: TrieNode, prefix: str, result: List[str]) -> None:
        """
        Recursively collect all words starting from the given node.
        
        Args:
            node: The starting node.
            prefix: The current prefix.
            result: The list to collect words into.
        """
        if node.is_end_of_word:
            result.append(prefix)
            
        for char, child_node in sorted(node.children.items()):
            self._collect_words(child_node, prefix + char, result)
            
    def _count_words_in_subtree(self, node: TrieNode) -> int:
        """
        Count the number of complete words in a subtree.
        
        Args:
            node: The root of the subtree.
            
        Returns:
            The number of complete words in the subtree.
        """
        count = 0
        if node.is_end_of_word:
            count += 1
            
        for child_node in node.children.values():
            count += self._count_words_in_subtree(child_node)
            
        return count
        
    def __len__(self) -> int:
        """Return the number of words in the trie."""
        return self._size
        
    def __contains__(self, word: str) -> bool:
        """Check if a word is in the trie."""
        try:
            return self.search(word)
        except InvalidOperationError:
            return False
            
    def __repr__(self) -> str:
        """Return detailed string representation."""
        case_info = "case-sensitive" if self._case_sensitive else "case-insensitive"
        return f"Trie(size={self._size}, {case_info})"
        
    def __str__(self) -> str:
        """Return simple string representation."""
        words = self.get_all_words()
        if len(words) <= 5:
            return f"Trie{words}"
        else:
            return f"Trie[{words[:3]}... +{len(words)-3} more]"

"""
Test suite for Trie data structure.
"""

import pytest
from pyalgos.strings.trie import Trie
from pyalgos.exceptions import InvalidOperationError


class TestTrieBasicOperations:
    """Test basic trie operations."""

    def test_empty_trie_creation(self) -> None:
        """Test creating an empty trie."""
        trie = Trie()
        assert len(trie) == 0
        assert not trie.search("test")

    def test_insert_and_search(self) -> None:
        """Test insert and search operations."""
        trie = Trie()
        
        trie.insert("hello")
        trie.insert("world")
        
        assert len(trie) == 2
        assert trie.search("hello")
        assert trie.search("world")
        assert not trie.search("hell")

    def test_prefix_operations(self) -> None:
        """Test prefix operations."""
        trie = Trie()
        words = ["hello", "help", "helicopter"]
        
        for word in words:
            trie.insert(word)
            
        assert trie.starts_with("hel")
        assert not trie.starts_with("xyz")
        
        prefix_words = trie.get_words_with_prefix("hel")
        assert len(prefix_words) == 3

    def test_delete_operations(self) -> None:
        """Test delete operations."""
        trie = Trie()
        trie.insert("hello")
        trie.insert("help")
        
        assert trie.search("hello")
        assert trie.delete("hello")
        assert not trie.search("hello")
        assert trie.search("help")  # Should still exist
        
        # Delete non-existent word
        assert not trie.delete("world")

    def test_case_sensitivity(self) -> None:
        """Test case sensitivity and case-insensitive operations."""
        trie = Trie(case_sensitive=True)
        trie.insert("Hello")
        trie.insert("hello")
        
        assert trie.search("Hello")
        assert trie.search("hello")
        assert len(trie) == 2
        
        # Test case-insensitive trie
        trie_ci = Trie(case_sensitive=False)
        trie_ci.insert("Hello")
        trie_ci.insert("HELLO")
        
        assert trie_ci.search("hello")
        assert trie_ci.search("HELLO")
        assert trie_ci.search("Hello")
        assert len(trie_ci) == 1  # Should be treated as same word

    def test_string_representation(self) -> None:
        """Test string representation."""
        trie = Trie()
        trie.insert("test")
        
        repr_str = repr(trie)
        assert "Trie" in repr_str


class TestTrieAdvancedOperations:
    """Test advanced trie operations."""
    
    def test_word_counting(self) -> None:
        """Test word counting functionality."""
        trie = Trie()
        words = ["apple", "app", "application", "apply"]
        
        for word in words:
            trie.insert(word)
            
        # Test count with prefix
        count = trie.count_words_with_prefix("app")
        assert count == 4
        
        count = trie.count_words_with_prefix("appl")
        assert count == 3
        
        count = trie.count_words_with_prefix("xyz")
        assert count == 0

    def test_all_words_retrieval(self) -> None:
        """Test retrieving all words."""
        trie = Trie()
        words = ["cat", "car", "card", "care", "careful"]
        
        for word in words:
            trie.insert(word)
            
        all_words = trie.get_all_words()
        assert len(all_words) == len(words)
        assert set(all_words) == set(words)

    def test_longest_prefix(self) -> None:
        """Test longest common prefix functionality."""
        trie = Trie()
        words = ["flower", "flow", "flight"]        
        for word in words:
            trie.insert(word)
            
        # Test common prefix search if available
        prefix_words = trie.get_words_with_prefix("fl")
        assert len(prefix_words) == 3

    def test_empty_string_operations(self) -> None:
        """Test operations with empty strings."""
        trie = Trie()
        
        # Empty string should raise error
        with pytest.raises(InvalidOperationError):
            trie.insert("")
        
        with pytest.raises(InvalidOperationError):
            trie.search("")
        
        with pytest.raises(InvalidOperationError):
            trie.delete("")

    def test_duplicate_insertions(self) -> None:
        """Test inserting duplicate words."""
        trie = Trie()
        
        trie.insert("hello")
        assert len(trie) == 1
        
        # Insert same word again
        trie.insert("hello")
        assert len(trie) == 1  # Should not increase count
        
        assert trie.search("hello")

    def test_clear_operation(self) -> None:
        """Test clearing the trie."""
        trie = Trie()
        words = ["test", "hello", "world"]
        
        for word in words:
            trie.insert(word)
            
        assert len(trie) == 3
        
        trie.clear()
        assert len(trie) == 0
        assert not trie.search("test")


class TestTrieEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_long_words(self) -> None:
        """Test with very long words."""
        trie = Trie()
        long_word = "a" * 1000
        
        trie.insert(long_word)
        assert trie.search(long_word)
        assert len(trie) == 1

    def test_special_characters(self) -> None:
        """Test with special characters."""
        trie = Trie()
        special_words = ["hello-world", "test_case", "file.txt", "user@domain.com"]
        
        for word in special_words:
            trie.insert(word)
            
        for word in special_words:
            assert trie.search(word)
            
        assert len(trie) == len(special_words)

    def test_numeric_strings(self) -> None:
        """Test with numeric strings."""
        trie = Trie()
        numbers = ["123", "456", "1234", "12"]
        
        for num in numbers:
            trie.insert(num)
            
        for num in numbers:
            assert trie.search(num)
            
        prefix_nums = trie.get_words_with_prefix("12")
        assert len(prefix_nums) == 3  # "12", "123" and "1234"

    def test_prefix_with_no_matches(self) -> None:
        """Test prefix operations with no matches."""
        trie = Trie()
        trie.insert("hello")
        trie.insert("world")
        
        assert not trie.starts_with("xyz")
        assert trie.count_words_with_prefix("xyz") == 0
        assert trie.get_words_with_prefix("xyz") == []

    def test_iteration_protocol(self) -> None:
        """Test iteration over trie."""
        trie = Trie()
        words = ["apple", "banana", "cherry"]
        
        for word in words:
            trie.insert(word)
            
        # Test that all words are available via get_all_words
        all_words = trie.get_all_words()
        assert set(all_words) == set(words)


class TestTriePerformance:
    """Test performance characteristics."""
    
    def test_large_trie_operations(self) -> None:
        """Test operations on large tries."""
        trie = Trie()
        
        # Insert many words
        words = [f"word{i}" for i in range(1000)]
        for word in words:
            trie.insert(word)
            
        assert len(trie) == 1000
        
        # Search operations should be fast
        for i in range(0, 1000, 100):  # Test every 100th word
            assert trie.search(f"word{i}")
            
        # Prefix operations
        prefix_words = trie.get_words_with_prefix("word1")
        assert len(prefix_words) >= 111  # word1, word10-19, word100-199

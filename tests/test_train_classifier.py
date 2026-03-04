"""Tests for the ML pipeline (models/train_classifier.py) and shared utilities."""

import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import tokenize


class TestTokenize:
    """Tests for the shared tokenize function."""

    def test_basic_tokenization(self):
        """Test that tokenize returns a list of tokens."""
        result = tokenize("This is a test message")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_lowercase(self):
        """Test that tokens are lowercased."""
        result = tokenize("HELLO WORLD")
        for token in result:
            assert token == token.lower()

    def test_punctuation_removal(self):
        """Test that punctuation is removed."""
        result = tokenize("Hello! How are you?")
        for token in result:
            assert token.isalnum(), f"Token '{token}' contains non-alphanumeric chars"

    def test_stopword_removal(self):
        """Test that common stopwords are removed."""
        result = tokenize("This is a very simple test")
        # 'this', 'is', 'a', 'very' are stopwords
        assert 'this' not in result
        assert 'is' not in result
        assert 'a' not in result

    def test_lemmatization(self):
        """Test that words are lemmatized."""
        result = tokenize("The dogs were running quickly")
        assert 'dog' in result

    def test_empty_string(self):
        """Test that empty string returns empty list."""
        result = tokenize("")
        assert result == []

    def test_disaster_message(self):
        """Test tokenization of a realistic disaster message."""
        message = "We need water and food supplies urgently!"
        result = tokenize(message)
        assert 'need' in result
        assert 'water' in result
        assert 'food' in result

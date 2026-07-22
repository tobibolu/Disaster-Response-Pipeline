"""Tests for the ML pipeline (models/train_classifier.py) and shared utilities."""

import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import utils
from utils import tokenize
from models.train_classifier import build_model, split_data


class TestTokenize:
    """Tests for the shared tokenize function."""

    def test_module_import_has_no_download_side_effects(self, monkeypatch):
        """Importing utils should not trigger network/resource downloads."""
        called = []

        def _record_download(*args, **kwargs):
            called.append(args[0] if args else kwargs.get("resource"))
            return True

        monkeypatch.setattr("nltk.download", _record_download)
        importlib.reload(utils)
        assert called == []

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

    def test_stemming(self):
        """Test that inflected words are reduced to a stable stem."""
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


def test_build_model_uses_multilabel_stratified_cross_validation():
    """Grid search should balance rare labels across its folds."""
    model = build_model()

    assert model.cv.__class__.__name__ == 'MultilabelStratifiedKFold'
    assert model.estimator.named_steps['vect'].token_pattern is None


def test_split_data_is_disjoint_and_preserves_rare_labels():
    """Every row appears once and both partitions retain each label."""
    X = pd.Series([f'message {index}' for index in range(20)])
    Y = pd.DataFrame({
        'related': [1] * 10 + [0] * 10,
        'water': [1 if index % 5 == 0 else 0 for index in range(20)],
    })

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.25)

    assert set(X_train.index).isdisjoint(X_test.index)
    assert sorted([*X_train.index, *X_test.index]) == list(range(20))
    assert np.all(Y_train.sum(axis=0).gt(0))
    assert np.all(Y_test.sum(axis=0).gt(0))

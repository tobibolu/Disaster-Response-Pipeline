"""Shared utilities for the Disaster Response Pipeline project."""

import re
from typing import List

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STEMMER = SnowballStemmer('english')


def tokenize(text: str) -> List[str]:
    """Tokenize, normalize, and stem a text string.

    Args:
        text: Raw text message to process.

    Returns:
        List of cleaned, stemmed tokens.
    """
    # Regex tokenization, sklearn's bundled stop-word set, and Snowball
    # stemming require no corpus downloads. Training and inference therefore
    # use exactly the same text transform in offline environments.
    words = TOKEN_PATTERN.findall(text.lower())
    return [
        STEMMER.stem(word)
        for word in words
        if word not in ENGLISH_STOP_WORDS
    ]

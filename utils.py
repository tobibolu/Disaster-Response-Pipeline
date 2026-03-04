"""Shared utilities for the Disaster Response Pipeline project."""

import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def tokenize(text: str) -> List[str]:
    """Tokenize, normalize, and lemmatize a text string.

    Args:
        text: Raw text message to process.

    Returns:
        List of cleaned, lemmatized tokens.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    return [
        LEMMATIZER.lemmatize(w).strip()
        for w in words
        if w not in STOP_WORDS
    ]

"""Shared utilities for the Disaster Response Pipeline project."""

import re
from threading import Lock
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOP_WORDS = set()
LEMMATIZER = WordNetLemmatizer()
_NLTK_READY = False
_NLTK_LOCK = Lock()


def _ensure_nltk_resources() -> None:
    """Load/download NLTK resources lazily at call time, not import time."""
    global STOP_WORDS, _NLTK_READY
    if _NLTK_READY:
        return

    with _NLTK_LOCK:
        if _NLTK_READY:
            return

        resources = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
        ]

        for resource_path, package in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                # Keep runtime robust in fresh/offline environments.
                nltk.download(package, quiet=True)

        try:
            STOP_WORDS = set(stopwords.words("english"))
        except LookupError:
            STOP_WORDS = set()

        _NLTK_READY = True


def tokenize(text: str) -> List[str]:
    """Tokenize, normalize, and lemmatize a text string.

    Args:
        text: Raw text message to process.

    Returns:
        List of cleaned, lemmatized tokens.
    """
    _ensure_nltk_resources()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    try:
        words = word_tokenize(text)
    except LookupError:
        words = text.split()

    tokens = []
    for word in words:
        if word in STOP_WORDS:
            continue
        try:
            token = LEMMATIZER.lemmatize(word).strip()
        except LookupError:
            token = word.strip()
        if token:
            tokens.append(token)
    return tokens

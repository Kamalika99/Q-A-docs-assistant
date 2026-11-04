"""Text analysis, normalization, sentence splitting, and Snowball stemming."""

import functools
import re
import unicodedata
from typing import List, Set, Tuple
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import ACTION_WORDS, GENERIC_TERMS, KEY_TERMS, SYNONYMS

STOPWORDS = set(ENGLISH_STOP_WORDS) | GENERIC_TERMS

# Cached stemmer
stem_word = functools.lru_cache(maxsize=None)(SnowballStemmer("english").stem)

_PHRASES = [
    (re.compile(r"\bsingle[\s-]+sign[\s-]?on\b", re.I), "sso"),
    (re.compile(r"\b(?:two|2|multi)[\s-]?(?:factor|step)\b(?:[\s-]+(?:authentication|verification))?", re.I), "2fa"),
    (re.compile(r"\b(?:log(?:g(?:ing|ed))?|logs|sign(?:ing|ed|s)?)[\s-]?in\b", re.I), "login"),
    (re.compile(r"\be-?mails?\b", re.I), "email"),
]

_ACTION_STEMS = {stem_word(w) for w in ACTION_WORDS}


def normalize_text(text: str) -> str:
    """Case-preserving clean-up: fixed phrases, contractions, thousands separators."""
    text = unicodedata.normalize("NFKC", text).replace("\u2019", "'")
    for pattern, replacement in _PHRASES:
        text = pattern.sub(replacement, text)
    text = re.sub(r"(?i)\b(\w+)n't\b", r"\1 not", text)
    text = re.sub(r"(?i)(?<=\w)'(?:s|re|ve|ll|d|m)\b", "", text)
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


def tokenize(text: str) -> List[str]:
    """Tokenize normalized text into lowercased alphanumeric tokens, applying synonyms."""
    tokens = re.findall(r"[a-z0-9]+", normalize_text(text).lower())
    tokens = [SYNONYMS.get(t, t) for t in tokens]
    return [t for t in tokens if len(t) >= 2 or t.isdigit()]


def content_terms(text: str) -> List[Tuple[str, str]]:
    """Return unique (word, stem) pairs for meaningful words (excluding stopwords/generic terms)."""
    seen, out = set(), []
    for tok in tokenize(text):
        if tok in STOPWORDS:
            continue
        stem = stem_word(tok)
        if stem not in seen:
            seen.add(stem)
            out.append((tok, stem))
    return out


def analyze_text(text: str) -> List[str]:
    """Analyzer for TfidfVectorizer: list of stems with stopwords removed."""
    return [stem_word(tok) for tok in tokenize(text) if tok not in STOPWORDS]


def important_terms(question: str, terms: List[Tuple[str, str]]) -> Set[str]:
    """Identify stems of named/key terms that the evidence must contain (T3).

    Includes acronyms (CSV, SSO), capitalised words that are not the first word
    (Growth, Salesforce), numbers, and anything in KEY_TERMS.
    """
    wanted = set()
    words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", normalize_text(question))
    for pos, word in enumerate(words):
        capitalised = pos > 0 and word[:1].isupper()
        acronym = len(word) >= 2 and word.isupper() and any(c.isalpha() for c in word)
        if capitalised or acronym:
            wanted.update(analyze_text(word))
    key_stems = {stem_word(t) for t in KEY_TERMS}
    for word, stem in terms:
        if stem in key_stems or word.isdigit():
            wanted.add(stem)
    return wanted & {stem for _, stem in terms}


def is_procedural(question: str) -> bool:
    """Check if the question asks for procedural instructions."""
    return bool(re.search(r"\b(?:how\s+(?:do|can|should|would|to)|steps?|instructions?|set\s*up|procedure)\b",
                          question, re.I))


def split_sentences(text: str) -> List[str]:
    """Robust sentence splitting respecting technical abbreviations and punctuation boundaries."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'`(\[])|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def has_action(sentence: str) -> bool:
    """Check if sentence contains action/procedural verbs."""
    return any(stem_word(tok) in _ACTION_STEMS for tok in tokenize(sentence))

"""Document loading, schema validation, and dual passage/document-level TF-IDF indexing."""

import functools
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DOCS_PATH
from nlp import analyze_text, has_action, split_sentences


class DocsLoadError(Exception):
    """Raised when the documentation file cannot be loaded or parsed."""


def load_documents(path=DOCS_PATH) -> Tuple[List[Dict[str, str]], List[str]]:
    """Load and validate articles. Returns (docs, warnings). Malformed entries are skipped."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DocsLoadError(f"Documentation file not found: {path}")
    except (OSError, ValueError) as exc:
        raise DocsLoadError(f"Documentation file could not be read: {exc}")

    items = raw.get("docs") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise DocsLoadError("Documentation file must contain a list of articles under 'docs'.")

    docs, warnings, seen = [], [], set()
    for n, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            warnings.append(f"Entry {n} skipped: not an object.")
            continue
        fields = {k: item.get(k) for k in ("doc_id", "title", "content")}
        if not all(isinstance(v, str) and v.strip() for v in fields.values()):
            warnings.append(f"Entry {n} skipped: doc_id, title and content must be non-empty text.")
            continue
        doc = {k: v.strip() for k, v in fields.items()}
        if doc["doc_id"] in seen:
            warnings.append(f"Entry {n} skipped: duplicate doc_id {doc['doc_id']}.")
            continue
        seen.add(doc["doc_id"])
        docs.append(doc)
    if not docs:
        raise DocsLoadError("No valid articles found in the documentation file.")
    return docs, warnings


def build_index(docs: List[Dict[str, str]], warnings: List[str] = None) -> Dict[str, Any]:
    """Build the TF-IDF index across both document text and fine-grained passages/sentences.

    This enables:
      1. Whole-document vector representation.
      2. Per-passage / per-sentence vector representations.
      3. Aggregated document scoring (combining doc-level and best-passage matching).
    """
    vectorizer = TfidfVectorizer(analyzer=analyze_text, sublinear_tf=True)
    doc_texts = [f"{d['title']}. {d['content']}" for d in docs]
    try:
        doc_matrix = vectorizer.fit_transform(doc_texts)
    except ValueError:
        raise DocsLoadError("Articles contain no usable words.")

    sentences: List[List[Dict[str, Any]]] = []
    sentence_matrices = []

    for d in docs:
        items = [
            {
                "text": s,
                "stems": frozenset(analyze_text(s)),
                "has_action": has_action(s),
            }
            for s in split_sentences(d["content"])
        ]
        sentences.append(items)
        # Vectorize sentences for passage-level similarity
        if items:
            s_matrix = vectorizer.transform([s["text"] for s in items])
        else:
            s_matrix = vectorizer.transform([d["title"]])
        sentence_matrices.append(s_matrix)

    return {
        "docs": docs,
        "warnings": list(warnings or []),
        "vectorizer": vectorizer,
        "matrix": doc_matrix,  # backward compatibility alias for doc_matrix
        "doc_matrix": doc_matrix,
        "vocab": set(vectorizer.vocabulary_),
        "doc_stems": [set(analyze_text(dt)) for dt in doc_texts],
        "sentences": sentences,
        "sentence_matrices": sentence_matrices,
    }


@functools.lru_cache(maxsize=4)
def _cached_index(path: str) -> Dict[str, Any]:
    return build_index(*load_documents(path))


def get_index(path=DOCS_PATH) -> Dict[str, Any]:
    """Return the cached TF-IDF index for the given document path."""
    return _cached_index(str(path))

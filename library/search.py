"""
PartSearchIndex: TF-IDF cosine similarity search over PartRepository records.

No external dependencies — uses only Python stdlib math and collections.
Build the index once with PartSearchIndex(repo), then call search() repeatedly.
Call build() to refresh after new parts are added to the repository.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from library.repository import PartKind, PartRecord, PartRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenisation, splitting on non-word characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _document_text(rec: PartRecord) -> str:
    """Concatenate all searchable fields into one string."""
    return " ".join([
        rec.name,
        rec.description,
        " ".join(rec.tags),
        rec.iso_standard or "",
    ])


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class PartSearchIndex:
    """
    TF-IDF index over all parts in a PartRepository.

    Usage::

        index = PartSearchIndex(repo)
        results = index.search("motor bracket aluminium", top_k=3)
        for record, score in results:
            print(record.name, f"{score:.3f}")

    Args:
        repository: PartRepository instance to index.
    """

    def __init__(self, repository: PartRepository) -> None:
        self._repo = repository
        self._records: list[PartRecord] = []
        self._tfidf: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        self.build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> None:
        """
        (Re)build the TF-IDF index from the current state of the repository.

        Call this after adding new parts to ensure fresh results.
        """
        self._records = self._repo.list_all()
        if not self._records:
            self._tfidf = []
            self._idf = {}
            return

        # Per-document normalised term frequencies
        tf_vectors: list[Counter] = []
        for rec in self._records:
            tokens = _tokenize(_document_text(rec))
            raw_tf = Counter(tokens)
            total = sum(raw_tf.values()) or 1
            tf_vectors.append(Counter({t: c / total for t, c in raw_tf.items()}))

        # Inverse document frequency (smoothed)
        N = len(self._records)
        df: Counter = Counter()
        for tf in tf_vectors:
            df.update(tf.keys())
        self._idf = {
            term: math.log((N + 1) / (count + 1)) + 1.0
            for term, count in df.items()
        }

        # TF-IDF weighted vectors
        self._tfidf = [
            {term: tf_val * self._idf.get(term, 1.0) for term, tf_val in tf.items()}
            for tf in tf_vectors
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.05,
        kind: PartKind | None = None,
    ) -> list[tuple[PartRecord, float]]:
        """
        Return up to top_k parts ranked by cosine similarity to the query.

        Args:
            query:     Free-text search string.
            top_k:     Maximum results to return.
            min_score: Cosine similarity threshold (0–1). Parts below this are excluded.
            kind:      If provided, restrict to CUSTOM or STANDARD parts.

        Returns:
            List of (PartRecord, score) sorted by score descending.
        """
        if not self._records:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        # Query TF-IDF vector
        q_raw_tf = Counter(q_tokens)
        q_total = sum(q_raw_tf.values())
        q_vec = {
            term: (count / q_total) * self._idf.get(term, 1.0)
            for term, count in q_raw_tf.items()
        }
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        results: list[tuple[PartRecord, float]] = []
        for rec, doc_vec in zip(self._records, self._tfidf):
            if kind is not None and rec.kind != kind:
                continue

            dot = sum(q_vec.get(term, 0.0) * val for term, val in doc_vec.items())
            doc_norm = math.sqrt(sum(v * v for v in doc_vec.values())) or 1.0
            score = dot / (q_norm * doc_norm)

            if score >= min_score:
                results.append((rec, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

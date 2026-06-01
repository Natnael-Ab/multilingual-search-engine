"""
Ranking algorithms: TF-IDF and BM25.

Both methods rank documents by how relevant they are to a query.
The difference is how they model term importance:

- TF-IDF: classic formula, intuitive, easy to explain
- BM25:   modern probabilistic model, handles document length better,
          generally outperforms TF-IDF in practice

We support both so the user can compare them side by side.
"""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .indexing import InvertedIndex


def rank_tfidf(query_tokens: list, index, doc_ids: list = None) -> list:
    """
    Rank documents using the TF-IDF scoring model.

    For each query term, we calculate:
        tf(t, d)  = raw frequency of term t in document d
        idf(t)    = log(N / df(t))   — how rare the term is
        score(d)  += tf * idf

    Documents with higher total scores are more relevant.

    Args:
        query_tokens: preprocessed query terms (may include duplicates for weighting)
        index:        InvertedIndex instance
        doc_ids:      optional subset of doc IDs to search (for language filtering)

    Returns:
        List of (doc_id, score) tuples sorted by score descending.
    """
    scores = {}

    # Use all indexed documents if no subset is given
    if doc_ids is None:
        doc_ids = set(index.get_all_doc_ids())
    else:
        doc_ids = set(doc_ids)

    for term in query_tokens:
        # Get IDF — how informative/rare this term is
        idf = index.get_idf(term)
        if idf == 0:
            continue  # term not in index, skip

        # Get all documents that contain this term
        postings = index.get_postings(term)

        for doc_id, tf in postings.items():
            if doc_id not in doc_ids:
                continue

            # TF-IDF score contribution for this term in this document
            # We use log-normalized TF to dampen the effect of very frequent terms
            tf_normalized = 1 + math.log(tf) if tf > 0 else 0
            tfidf = tf_normalized * idf

            scores[doc_id] = scores.get(doc_id, 0.0) + tfidf

    # Sort by score, highest first
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def rank_bm25(query_tokens: list, index, doc_ids: list = None,
              k1: float = 1.5, b: float = 0.75) -> list:
    """
    Rank documents using the BM25 (Best Match 25) scoring model.

    BM25 improves on TF-IDF by:
      1. Saturating term frequency — doubling TF doesn't double the score
      2. Normalizing by document length — longer docs don't unfairly dominate

    Formula for each term t in query, document d:
        idf(t)  = log((N - df + 0.5) / (df + 0.5))
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        score   += idf * tf_norm

    Parameters:
        k1:  term saturation parameter — typically 1.2–2.0
        b:   length normalization — 0 = no normalization, 1 = full normalization

    Args:
        query_tokens: preprocessed query terms
        index:        InvertedIndex instance
        doc_ids:      optional subset of doc IDs to search
        k1, b:        BM25 tuning parameters

    Returns:
        List of (doc_id, score) tuples sorted by score descending.
    """
    scores = {}
    avg_dl = index.avg_doc_length

    if doc_ids is None:
        doc_ids = set(index.get_all_doc_ids())
    else:
        doc_ids = set(doc_ids)

    for term in query_tokens:
        # IDF component — rarer terms get higher weight
        df = index.df.get(term, 0)
        if df == 0:
            continue  # term not in index

        N = index.num_docs
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

        # TF component with length normalization
        postings = index.get_postings(term)

        for doc_id, tf in postings.items():
            if doc_id not in doc_ids:
                continue

            dl = index.get_doc_length(doc_id)

            # BM25 TF saturation — prevents very frequent terms from dominating
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_dl if avg_dl > 0 else 1)))

            scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_norm

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def rank_documents(query_tokens: list, method: str, index,
                   doc_ids: list = None) -> list:
    """
    Dispatch to the correct ranking function based on the method name.

    Args:
        query_tokens: preprocessed query terms
        method:       'tfidf' or 'bm25'
        index:        InvertedIndex instance
        doc_ids:      optional list of doc IDs to restrict the search

    Returns:
        Ranked list of (doc_id, score) tuples.
    """
    method = method.lower().strip()

    if method == 'bm25':
        return rank_bm25(query_tokens, index, doc_ids)
    else:
        # Default to TF-IDF
        return rank_tfidf(query_tokens, index, doc_ids)

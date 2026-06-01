"""
IR evaluation metrics: Precision, Recall, F1, and Precision@k.

Evaluation tells us how good our retrieval system actually is.
We compare what the system retrieved (retrieved set) against
what the correct answers are (relevant set) from a ground truth.

This module also contains pre-defined relevance judgments for our
small document collection — a set of test queries with their
known-relevant documents. In a real IR lab, these would come from
human annotators.
"""


# ── Ground truth relevance judgments ─────────────────────────────────────────
# Format: query → list of relevant document IDs
# These were manually assigned based on document content
RELEVANCE_JUDGMENTS = {
    "health technology": [
        "en_01", "en_11", "en_12", "am_01", "am_11"
    ],
    "doctor hospital disease": [
        "en_01", "en_11", "am_01", "am_11"
    ],
    "education university learning": [
        "en_03", "en_08", "am_03", "am_08"
    ],
    "agriculture farming food": [
        "en_06", "en_13", "am_06", "am_13"
    ],
    "technology computer internet": [
        "en_02", "en_12", "am_02", "am_12"
    ],
    "environment climate water": [
        "en_04", "am_04"
    ],
    "economy market business": [
        "en_07", "am_07"
    ],
    "sport football athlete": [
        "en_05", "am_05"
    ],
    "culture tradition art": [
        "en_09", "am_09"
    ],
    "government democracy election": [
        "en_10", "am_10"
    ],
    "ጤና ሐኪም ሆስፒታል": [
        "am_01", "am_11", "en_01", "en_11"
    ],
    "ትምህርት ዩኒቨርሲቲ ተማሪ": [
        "am_03", "am_08", "en_03", "en_08"
    ],
    "ቴክኖሎጂ ኮምፒዩተር ኢንተርኔት": [
        "am_02", "am_12", "en_02", "en_12"
    ],
    "ግብርና እርሻ ምግብ": [
        "am_06", "am_13", "en_06", "en_13"
    ],
    "ስፖርት እግር ኳስ ውድድር": [
        "am_05", "en_05"
    ],
}


def precision(retrieved: list, relevant: list) -> float:
    """
    Precision = |Retrieved ∩ Relevant| / |Retrieved|

    Of all the documents we returned, what fraction were actually relevant?
    High precision means fewer false positives (irrelevant results).
    """
    if not retrieved:
        return 0.0
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    return len(retrieved_set & relevant_set) / len(retrieved_set)


def recall(retrieved: list, relevant: list) -> float:
    """
    Recall = |Retrieved ∩ Relevant| / |Relevant|

    Of all the relevant documents that exist, how many did we find?
    High recall means fewer false negatives (missed relevant docs).
    """
    if not relevant:
        return 0.0
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def f1_score(p: float, r: float) -> float:
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    The harmonic mean of precision and recall — a single number that
    balances both. Useful when you care equally about both metrics.
    """
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def precision_at_k(retrieved: list, relevant: list, k: int = 5) -> float:
    """
    P@k = |top-k Retrieved ∩ Relevant| / k

    Precision of just the top-k results. Important because users
    typically only look at the first page of results.
    """
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc in top_k if doc in relevant_set)
    return hits / k


def average_precision(retrieved: list, relevant: list) -> float:
    """
    Average Precision (AP) — considers the rank of each relevant document.

    We calculate precision at each rank where we find a relevant doc,
    then average those precision values. Rewards systems that rank
    relevant docs higher.
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant_set) if relevant_set else 0.0


def evaluate_all_queries(search_fn, ranking_method: str = 'bm25') -> list:
    """
    Run all test queries and compute evaluation metrics for each.

    This is used by the evaluation dashboard to show how well the
    system performs on a standardized set of queries.

    Args:
        search_fn:      function that takes (query, method) and returns list of doc IDs
        ranking_method: 'tfidf' or 'bm25'

    Returns:
        List of result dicts with all metrics for each query.
    """
    results = []

    for query, relevant_docs in RELEVANCE_JUDGMENTS.items():
        try:
            # Run the actual search
            retrieved_docs = search_fn(query, ranking_method)
        except Exception:
            retrieved_docs = []

        # Calculate all metrics
        p   = precision(retrieved_docs, relevant_docs)
        r   = recall(retrieved_docs, relevant_docs)
        f1  = f1_score(p, r)
        p5  = precision_at_k(retrieved_docs, relevant_docs, k=5)
        ap  = average_precision(retrieved_docs, relevant_docs)

        results.append({
            'query':          query,
            'retrieved':      len(retrieved_docs),
            'relevant_total': len(relevant_docs),
            'hits':           len(set(retrieved_docs) & set(relevant_docs)),
            'precision':      round(p, 4),
            'recall':         round(r, 4),
            'f1':             round(f1, 4),
            'precision_at_5': round(p5, 4),
            'avg_precision':  round(ap, 4),
            'ranking_method': ranking_method,
        })

    return results


def summarize_evaluation(eval_results: list) -> dict:
    """
    Compute macro-averaged metrics over all queries.

    Macro-averaging treats each query equally, regardless of how
    many relevant documents it has.
    """
    if not eval_results:
        return {}

    n = len(eval_results)
    return {
        'map':               round(sum(r['avg_precision']  for r in eval_results) / n, 4),
        'mean_precision':    round(sum(r['precision']      for r in eval_results) / n, 4),
        'mean_recall':       round(sum(r['recall']         for r in eval_results) / n, 4),
        'mean_f1':           round(sum(r['f1']             for r in eval_results) / n, 4),
        'mean_precision_at_5': round(sum(r['precision_at_5'] for r in eval_results) / n, 4),
        'total_queries':     n,
    }

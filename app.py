"""
app.py — Main Flask application for the Multilingual IR System.

Routes:
    /                   — home / search page
    /search             — execute a search and show results
    /document/<doc_id>  — full document view with highlighted terms
    /evaluation         — evaluation dashboard (Precision, Recall, F1, P@5)
    /autocomplete       — JSON endpoint for search bar suggestions
    /api/stats          — JSON endpoint for index statistics

Run:
    python app.py
"""

import os
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify

from ir.search_engine import SearchEngine
from ir.evaluation    import evaluate_all_queries, summarize_evaluation

# ── Application Setup ─────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'multilingual-ir-secret-2024')

# Path to the dataset directory (relative to this file)
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')

# Initialize the search engine once at startup — it builds (or loads) the index
print("[app] Initializing search engine...")
engine = SearchEngine(DATASET_DIR)
print(f"[app] Search engine ready — {engine.index.num_docs} documents indexed")


# ── Helper: keep track of recent searches ─────────────────────────────────────
# We store at most 10 recent searches in a simple in-memory list.
# (In a real system this would be per-user and stored in a session or DB.)
_recent_searches = []

def _add_recent_search(query: str):
    global _recent_searches
    if query and query not in _recent_searches:
        _recent_searches.insert(0, query)
    _recent_searches = _recent_searches[:10]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """
    Home / search page.
    Shows the search bar, language selector, and recent searches.
    """
    stats = engine.get_stats()
    return render_template(
        'index.html',
        stats=stats,
        recent_searches=_recent_searches[:5],
    )


@app.route('/search')
def search():
    """
    Search results page.

    Reads query parameters from the URL:
        q         — the raw query string
        lang      — 'auto', 'english', or 'amharic'  (default: auto)
        ranking   — 'bm25' or 'tfidf'                (default: bm25)
        expand    — '1' to enable query expansion     (default: 1)
        page      — result page number               (default: 1)
    """
    query          = request.args.get('q', '').strip()
    ranking_method = request.args.get('ranking', 'bm25').lower()
    use_expansion  = request.args.get('expand', '1') == '1'
    page           = int(request.args.get('page', 1))
    per_page       = 10

    # If the query is empty just redirect home
    if not query:
        return redirect(url_for('index'))

    # Keep track of what people search for
    _add_recent_search(query)

    # Run the search
    result = engine.search(
        query          = query,
        ranking_method = ranking_method,
        use_expansion  = use_expansion,
        page           = page,
        per_page       = per_page,
    )

    stats = engine.get_stats()

    return render_template(
        'results.html',
        query          = query,
        ranking_method = ranking_method,
        use_expansion  = use_expansion,
        results        = result['results'],
        total          = result['total'],
        page           = page,
        per_page       = per_page,
        total_pages    = result['total_pages'],
        query_info     = result['query_info'],
        search_stats   = result['stats'],
        stats          = stats,
    )


@app.route('/document/<doc_id>')
def document(doc_id: str):
    """
    Full document view.

    Shows the complete document content with query terms highlighted
    using <mark> tags. Also shows related documents.

    URL params:
        q        — original query (for highlighting and back-link)
        ranking  — 'bm25' or 'tfidf'
    """
    query          = request.args.get('q', '').strip()
    ranking_method = request.args.get('ranking', 'bm25').lower()

    doc_data = engine.get_document(doc_id, query=query, ranking_method=ranking_method)

    if doc_data is None:
        return render_template('error.html',
                               error_code=404,
                               message=f'Document "{doc_id}" not found.'), 404

    return render_template(
        'document.html',
        doc            = doc_data,
        query          = query,
        ranking_method = ranking_method,
    )


@app.route('/evaluation')
def evaluation():
    """
    Evaluation dashboard — shows Precision, Recall, F1, P@5 for
    a set of ground-truth queries, comparing TF-IDF vs BM25.
    """
    # Run evaluation for both ranking methods
    tfidf_results = evaluate_all_queries(engine.search_for_eval, 'tfidf')
    bm25_results  = evaluate_all_queries(engine.search_for_eval, 'bm25')

    tfidf_summary = summarize_evaluation(tfidf_results)
    bm25_summary  = summarize_evaluation(bm25_results)

    stats = engine.get_stats()

    return render_template(
        'evaluation.html',
        tfidf_results  = tfidf_results,
        bm25_results   = bm25_results,
        tfidf_summary  = tfidf_summary,
        bm25_summary   = bm25_summary,
        stats          = stats,
    )


@app.route('/autocomplete')
def autocomplete():
    """
    JSON endpoint that returns autocomplete suggestions.
    Called by the search bar JavaScript as the user types.
    """
    partial = request.args.get('q', '').strip()
    suggestions = engine.autocomplete(partial)
    return jsonify({'suggestions': suggestions})


@app.route('/api/stats')
def api_stats():
    """
    JSON endpoint with index and system statistics.
    Useful for the status bar and for debugging.
    """
    stats = engine.get_stats()
    return jsonify(stats)


# ── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error_code=404,
                           message='Page not found.'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error_code=500,
                           message='Something went wrong on our end. Please try again.'), 500


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Debug mode makes development easier — shows errors in the browser
    app.run(host='0.0.0.0', port=port, debug=False)

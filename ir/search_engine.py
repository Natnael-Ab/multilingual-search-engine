"""
SearchEngine — the central coordinator for the IR system.

This class ties together all the IR components:
  - Document loading from the dataset directory
  - Text preprocessing (English + Amharic)
  - Index building and caching
  - Query expansion (optional)
  - Cross-lingual translation (for multilingual retrieval)
  - TF-IDF and BM25 ranking
  - Full document viewing with query term highlighting
  - Related document suggestions

The engine indexes documents once at startup (or loads a saved index)
and then answers queries in real time.
"""

import os
import time
import re
import math

from .preprocessing   import preprocess, detect_language
from .indexing        import InvertedIndex
from .ranking         import rank_documents
from .translation     import get_cross_lingual_terms, translate_terms
from .query_expansion import expand_query, get_suggestions

# Where we persist the index between server restarts
INDEX_FILE = os.path.join(os.path.dirname(__file__), '..', 'saved_indexes', 'main_index.json')


class SearchEngine:
    """
    Main IR system — initialize once, query many times.
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.index = InvertedIndex()
        self._build_or_load_index()

    # ── Initialization ─────────────────────────────────────────────────────────

    def _build_or_load_index(self):
        """
        Try to load a saved index; rebuild from scratch if none exists.

        Rebuilding scans the dataset/ directory, preprocesses every
        document, and constructs the inverted index.
        """
        loaded = self.index.load(INDEX_FILE)
        if loaded:
            print(f"[SearchEngine] Loaded index: {self.index.num_docs} documents")
            return

        print("[SearchEngine] Building index from dataset...")
        documents = self._load_documents()
        self.index.build(documents)
        self.index.save(INDEX_FILE)
        print(f"[SearchEngine] Index built: {self.index.num_docs} documents, "
              f"{len(self.index.index)} unique terms")

    def _load_documents(self) -> list:
        """
        Read all .txt files from dataset/english/ and dataset/amharic/.

        Each file name is expected to be like:  doc_id__Title__Category.txt
        If the name has no separators we use the filename as the title.
        """
        documents = []

        for lang_folder in ['english', 'amharic']:
            folder = os.path.join(self.dataset_dir, lang_folder)
            if not os.path.isdir(folder):
                continue

            for filename in sorted(os.listdir(folder)):
                if not filename.endswith('.txt'):
                    continue

                filepath = os.path.join(folder, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                # Parse metadata from the first lines of the file
                # Format: first line = title, second line = category, rest = body
                lines = content.split('\n')
                title    = lines[0].strip()   if len(lines) > 0 else filename
                category = lines[1].strip()   if len(lines) > 1 else 'General'
                body     = '\n'.join(lines[2:]).strip() if len(lines) > 2 else content

                # Use filename (without extension) as the document ID
                doc_id = filename.replace('.txt', '')

                # Preprocess the body text into tokens for indexing
                # lang_folder is 'english' or 'amharic' — pass it directly
                tokens = preprocess(body, lang_folder)

                documents.append({
                    'id':       doc_id,
                    'title':    title,
                    'language': lang_folder,   # 'english' or 'amharic'
                    'category': category,
                    'content':  body,
                    'tokens':   tokens,
                })

        return documents

    def rebuild_index(self):
        """Force a full index rebuild — useful if the dataset changes."""
        documents = self._load_documents()
        self.index.build(documents)
        self.index.save(INDEX_FILE)

    # ── Core Search ────────────────────────────────────────────────────────────

    def search(self, query: str, ranking_method: str = 'bm25',
               use_expansion: bool = True, page: int = 1,
               per_page: int = 10) -> dict:
        """
        Execute a multilingual search and return paginated results.

        Steps:
          1. Detect query language
          2. Preprocess query tokens
          3. Optionally expand with synonyms
          4. Also translate query to the other language (cross-lingual)
          5. Rank each language's document set separately
          6. Merge and return combined ranked results

        Args:
            query:          raw user query string
            ranking_method: 'tfidf' or 'bm25'
            use_expansion:  whether to add synonym-based terms
            page:           result page number (1-indexed)
            per_page:       number of results per page

        Returns dict with: results, total, page, total_pages, stats, query_info
        """
        start_time = time.time()

        if not query.strip():
            return self._empty_result(query)

        # Step 1 — detect query language
        query_lang = detect_language(query)

        # Step 2 — preprocess query in its own language
        query_tokens = preprocess(query, query_lang)

        if not query_tokens:
            return self._empty_result(query)

        # Step 3 — optionally expand with synonyms
        if use_expansion:
            query_tokens = expand_query(query_tokens, query_lang)

        # Step 4 — translate query terms to the other language (cross-lingual)
        cross_tokens = get_cross_lingual_terms(query, query_lang, preprocess)
        if use_expansion and cross_tokens:
            cross_tokens = expand_query(cross_tokens,
                                        'amharic' if query_lang == 'english' else 'english')

        # Step 5 — identify which docs belong to each language
        en_doc_ids = [d for d in self.index.get_all_doc_ids()
                      if self.index.get_document(d)['language'] == 'english']
        am_doc_ids = [d for d in self.index.get_all_doc_ids()
                      if self.index.get_document(d)['language'] == 'amharic']

        # Rank same-language documents with original query tokens
        if query_lang == 'english':
            same_lang_results = rank_documents(query_tokens, ranking_method,
                                               self.index, en_doc_ids)
            cross_lang_results = rank_documents(cross_tokens or query_tokens,
                                                ranking_method, self.index, am_doc_ids)
        else:
            same_lang_results = rank_documents(query_tokens, ranking_method,
                                               self.index, am_doc_ids)
            cross_lang_results = rank_documents(cross_tokens or query_tokens,
                                                ranking_method, self.index, en_doc_ids)

        # Step 6 — merge both result sets
        # We interleave them: take pairs of (same-lang, cross-lang) for balance
        all_results = self._merge_results(same_lang_results, cross_lang_results)

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        # Paginate
        total = len(all_results)
        start = (page - 1) * per_page
        end   = start + per_page
        page_results = all_results[start:end]

        # Build result cards with snippets and metadata
        result_cards = []
        for doc_id, score in page_results:
            doc = self.index.get_document(doc_id)
            if not doc:
                continue
            snippet = self._make_snippet(doc['content'], query_tokens + cross_tokens, 200)
            result_cards.append({
                'id':             doc_id,
                'title':          doc['title'],
                'language':       doc['language'],
                'category':       doc['category'],
                'snippet':        snippet,
                'score':          round(score, 4),
                'ranking_method': ranking_method.upper(),
            })

        return {
            'results':       result_cards,
            'total':         total,
            'page':          page,
            'per_page':      per_page,
            'total_pages':   math.ceil(total / per_page) if total > 0 else 0,
            'query_info': {
                'original_query':  query,
                'query_language':  query_lang,
                'query_tokens':    query_tokens,
                'cross_tokens':    cross_tokens,
                'expansion_used':  use_expansion,
                'ranking_method':  ranking_method,
            },
            'stats': {
                'retrieval_time_ms': elapsed_ms,
                'total_results':     total,
                'documents_indexed': self.index.num_docs,
            },
        }

    def _merge_results(self, same_lang: list, cross_lang: list) -> list:
        """
        Combine same-language and cross-language results into one ranked list.

        Strategy: normalize scores within each group to [0, 1], then
        slightly weight same-language results higher (they usually match better).
        After normalizing, we can compare and interleave them fairly.
        """
        def normalize(results):
            if not results:
                return []
            max_score = results[0][1] if results else 1
            if max_score == 0:
                max_score = 1
            return [(doc_id, score / max_score) for doc_id, score in results]

        norm_same  = normalize(same_lang)
        norm_cross = normalize(cross_lang)

        # Boost same-language results by 20%
        boosted_same  = [(d, s * 1.2) for d, s in norm_same]
        boosted_cross = [(d, s * 1.0) for d, s in norm_cross]

        # Merge into single list and re-sort
        merged = boosted_same + boosted_cross
        merged.sort(key=lambda x: x[1], reverse=True)

        return merged

    def _make_snippet(self, content: str, query_tokens: list,
                      max_len: int = 200) -> str:
        """
        Extract a relevant excerpt from the document body.

        We try to find a sentence that contains one of the query terms.
        If none found, we fall back to the start of the document.
        The snippet is then HTML-safe (we don't add <mark> here —
        that happens in the template with highlight_terms).
        """
        sentences = re.split(r'[.!?\n]+', content)
        query_set = set(t.lower() for t in query_tokens)

        # Find the best sentence that overlaps with query terms
        for sentence in sentences:
            words = set(sentence.lower().split())
            if words & query_set:
                snip = sentence.strip()[:max_len]
                return snip + '...' if len(sentence.strip()) > max_len else snip

        # Fallback to first 200 characters
        return content[:max_len].strip() + '...'

    # ── Document Viewing ───────────────────────────────────────────────────────

    def get_document(self, doc_id: str, query: str = None,
                     ranking_method: str = 'bm25') -> dict:
        """
        Return the full document data for the document view page.

        If a query is supplied, we compute the relevance score and
        highlight query terms in the content for the reader.

        Returns:
            dict with all document fields plus score, highlighted_content,
            and related_documents.
        """
        doc = self.index.get_document(doc_id)
        if not doc:
            return None

        # Compute relevance score against the query (if given)
        score = 0.0
        highlighted_content = doc['content']
        query_tokens = []
        cross_tokens  = []

        if query:
            query_lang   = detect_language(query)
            query_tokens = preprocess(query, query_lang)
            cross_tokens = get_cross_lingual_terms(query, query_lang, preprocess)

            all_tokens = list(set(query_tokens + cross_tokens))

            # Rank just this one doc to get its score
            ranked = rank_documents(all_tokens, ranking_method,
                                    self.index, [doc_id])
            score = ranked[0][1] if ranked else 0.0

            # Highlight query terms in the document body
            highlighted_content = highlight_terms(doc['content'], query_tokens + cross_tokens)

        # Find related documents (same category, same language)
        related = self._get_related(doc_id, doc['category'], doc['language'], n=5)

        return {
            'id':                  doc['id'],
            'title':               doc['title'],
            'language':            doc['language'],
            'category':            doc['category'],
            'content':             doc['content'],
            'highlighted_content': highlighted_content,
            'score':               round(score, 4),
            'ranking_method':      ranking_method.upper(),
            'query':               query or '',
            'query_tokens':        query_tokens,
            'related_documents':   related,
        }

    def _get_related(self, doc_id: str, category: str,
                     language: str, n: int = 5) -> list:
        """
        Find documents related to a given document by category and language.

        We first look for same-category same-language docs, then
        broaden to same-category cross-language docs.
        """
        related = []

        for other_id, other_doc in self.index.documents.items():
            if other_id == doc_id:
                continue
            if other_doc['category'] == category:
                related.append({
                    'id':       other_doc['id'],
                    'title':    other_doc['title'],
                    'language': other_doc['language'],
                    'category': other_doc['category'],
                })
            if len(related) >= n:
                break

        return related

    # ── Autocomplete ───────────────────────────────────────────────────────────

    def autocomplete(self, partial: str) -> list:
        """Return up to 8 query suggestions matching the partial input."""
        lang = detect_language(partial)
        return get_suggestions(partial, lang)

    # ── Evaluation helper ──────────────────────────────────────────────────────

    def search_for_eval(self, query: str, ranking_method: str = 'bm25') -> list:
        """
        Run a search and return just the list of doc IDs (for evaluation).
        Returns at most 20 results (standard for IR evaluation).
        """
        result = self.search(query, ranking_method, use_expansion=False,
                             page=1, per_page=20)
        return [r['id'] for r in result['results']]

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return index statistics for the UI status bar."""
        return self.index.stats()

    def _empty_result(self, query: str) -> dict:
        return {
            'results': [], 'total': 0, 'page': 1,
            'per_page': 10, 'total_pages': 0,
            'query_info': {'original_query': query},
            'stats': {'retrieval_time_ms': 0,
                      'total_results': 0,
                      'documents_indexed': self.index.num_docs},
        }


def highlight_terms(text: str, tokens: list) -> str:
    """
    Wrap occurrences of query terms in <mark>...</mark> tags.

    This is used in the document view page so users can quickly
    see where their query terms appear in the full document text.

    We highlight whole words only (not substrings inside longer words).
    Amharic: we use a simpler substring match since word boundaries
    aren't well defined with the Ethiopic script.
    """
    if not tokens:
        return text

    for token in set(tokens):
        if not token:
            continue
        # Check if it's Amharic (Ethiopic characters)
        is_ethiopic = any('\u1200' <= c <= '\u137F' for c in token)

        if is_ethiopic:
            # Simple substring replacement for Amharic
            text = text.replace(token, f'<mark>{token}</mark>')
        else:
            # Word-boundary aware replacement for English
            pattern = re.compile(r'\b' + re.escape(token) + r'\b', re.IGNORECASE)
            text = pattern.sub(lambda m: f'<mark>{m.group()}</mark>', text)

    return text

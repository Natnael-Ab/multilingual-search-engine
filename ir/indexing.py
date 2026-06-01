"""
Inverted index construction and management.

The inverted index is the core data structure of any IR system.
For each term, it maps to the list of documents containing that term,
along with frequency counts. This lets us quickly find which documents
contain a query term without scanning every document.

Structure:
    {
        "term": {
            "doc1": frequency,
            "doc3": frequency,
            ...
        },
        ...
    }

We also store:
    - document frequency (df): how many docs contain the term
    - document lengths: number of tokens in each document
    - raw document metadata: title, language, category, content
"""

import os
import json
import math
from collections import defaultdict


class InvertedIndex:
    """
    Builds and stores an inverted index over a document collection.

    Usage:
        idx = InvertedIndex()
        idx.build(documents)
        idx.save('saved_indexes/index.json')
        # Later:
        idx.load('saved_indexes/index.json')
    """

    def __init__(self):
        # Main inverted index: term → {doc_id: term_frequency}
        self.index = defaultdict(dict)

        # Document frequency: term → number of docs containing it
        self.df = {}

        # Document lengths: doc_id → token count (used for BM25)
        self.doc_lengths = {}

        # Average document length across the collection
        self.avg_doc_length = 0.0

        # Total number of indexed documents
        self.num_docs = 0

        # Document metadata: doc_id → {title, language, category, content, tokens}
        self.documents = {}

    def build(self, documents: list):
        """
        Build the inverted index from a list of document dicts.

        Each document should have:
            id, title, language, category, content, tokens (preprocessed)
        """
        self.num_docs = len(documents)
        self.index = defaultdict(dict)

        for doc in documents:
            doc_id = doc['id']

            # Store document metadata for later retrieval
            self.documents[doc_id] = {
                'id':       doc['id'],
                'title':    doc['title'],
                'language': doc['language'],
                'category': doc['category'],
                'content':  doc['content'],
                'tokens':   doc.get('tokens', []),
            }

            tokens = doc.get('tokens', [])

            # Count how often each term appears in this document (term frequency)
            term_counts = defaultdict(int)
            for token in tokens:
                term_counts[token] += 1

            # Store the document length (total token count)
            self.doc_lengths[doc_id] = len(tokens)

            # Add each term's frequency to the inverted index
            for term, freq in term_counts.items():
                self.index[term][doc_id] = freq

        # Calculate document frequencies: for each term, how many docs contain it
        self.df = {term: len(postings) for term, postings in self.index.items()}

        # Calculate average document length for BM25 normalization
        if self.doc_lengths:
            total_tokens = sum(self.doc_lengths.values())
            self.avg_doc_length = total_tokens / self.num_docs
        else:
            self.avg_doc_length = 0.0

    def get_postings(self, term: str) -> dict:
        """
        Return the posting list for a term: {doc_id: term_frequency}.
        Returns empty dict if the term is not in the index.
        """
        return dict(self.index.get(term, {}))

    def get_idf(self, term: str) -> float:
        """
        Compute the Inverse Document Frequency for a term.

        IDF = log((N - df + 0.5) / (df + 0.5))
        where N = total documents, df = document frequency of the term.

        High IDF means the term is rare → more discriminative.
        """
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        # Smooth IDF formula (BM25 style) — avoids zero division
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

    def get_tf(self, term: str, doc_id: str) -> int:
        """Return the raw term frequency of a term in a specific document."""
        return self.index.get(term, {}).get(doc_id, 0)

    def get_doc_length(self, doc_id: str) -> int:
        """Return the token count for a document (used in BM25 normalization)."""
        return self.doc_lengths.get(doc_id, 0)

    def get_all_doc_ids(self) -> list:
        """Return all document IDs in the collection."""
        return list(self.documents.keys())

    def get_document(self, doc_id: str) -> dict:
        """Return the metadata dict for a specific document."""
        return self.documents.get(doc_id)

    def save(self, filepath: str):
        """
        Persist the index to a JSON file so we don't have to
        rebuild it on every server restart.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            'index':          dict(self.index),
            'df':             self.df,
            'doc_lengths':    self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'num_docs':       self.num_docs,
            'documents':      self.documents,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str) -> bool:
        """
        Load a previously saved index from disk.

        Returns True on success, False if the file doesn't exist yet.
        """
        if not os.path.exists(filepath):
            return False

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.index          = defaultdict(dict, data.get('index', {}))
        self.df             = data.get('df', {})
        self.doc_lengths    = data.get('doc_lengths', {})
        self.avg_doc_length = data.get('avg_doc_length', 0.0)
        self.num_docs       = data.get('num_docs', 0)
        self.documents      = data.get('documents', {})

        return True

    def stats(self) -> dict:
        """Return a summary of the index statistics."""
        return {
            'total_documents':  self.num_docs,
            'unique_terms':     len(self.index),
            'avg_doc_length':   round(self.avg_doc_length, 2),
            'english_docs':     sum(1 for d in self.documents.values() if d['language'] == 'english'),
            'amharic_docs':     sum(1 for d in self.documents.values() if d['language'] == 'amharic'),
        }

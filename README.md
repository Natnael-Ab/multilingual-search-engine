# LexisSearch — Multilingual Information Retrieval System

A bilingual IR system supporting both **English** and **Amharic** search, built as a university academic project. The system retrieves relevant documents from both language collections regardless of query language.

---

## Overview

LexisSearch is a Flask-based search engine that demonstrates core Information Retrieval concepts applied to a real bilingual document collection. Users can type a query in English or Amharic and retrieve ranked results from both language corpora.

**Key features:**
- **Cross-lingual retrieval** — English queries find Amharic documents, and vice versa
- **Dual ranking** — TF-IDF and BM25 selectable from the UI
- **Query expansion** — synonym-based expansion to improve recall
- **Full document view** — click any result to open the complete document with highlighted query terms
- **Evaluation dashboard** — Precision, Recall, F1, P@5, MAP for both ranking methods

---

## Project Structure

```
flask-ir/
├── app.py                  ← Main Flask application, all routes
├── requirements.txt        ← Python dependencies
│
├── ir/
│   ├── preprocessing.py    ← English (Porter stemmer) + Amharic (suffix strip) preprocessing
│   ├── indexing.py         ← Inverted index: {term: {doc_id: tf}} with df and doc lengths
│   ├── ranking.py          ← TF-IDF and BM25 scoring functions
│   ├── translation.py      ← Bilingual English↔Amharic dictionary for CLIR
│   ├── query_expansion.py  ← Synonym-based query expansion + autocomplete
│   ├── evaluation.py       ← Precision, Recall, F1, P@5, MAP metrics
│   └── search_engine.py    ← SearchEngine class — orchestrates all IR components
│
├── dataset/
│   ├── english/            ← 13 English documents (.txt)
│   └── amharic/            ← 13 Amharic documents (.txt)
│
├── templates/
│   ├── base.html           ← Shared layout (navbar, footer, dark mode)
│   ├── index.html          ← Home / search page
│   ├── results.html        ← Search results with pagination
│   ├── document.html       ← Full document view with highlighted terms
│   ├── evaluation.html     ← Evaluation dashboard
│   └── error.html          ← 404 / 500 error pages
│
├── static/
│   ├── css/style.css       ← Custom styles (glassmorphism, animations)
│   └── js/main.js          ← Theme toggle, autocomplete, UI interactions
│
└── saved_indexes/
    └── main_index.json     ← Persisted inverted index (auto-generated)
```

---

## Setup & Running

```bash
# Install Python packages
pip install flask nltk flask-cors

# Run the app
python app.py
```

The server starts at `http://localhost:5000`.

---

## IR Architecture

### 1. Document Preprocessing

**English:**
- Lowercase normalization
- Punctuation removal (regex)
- Stopword removal (NLTK English stopwords)
- Porter stemming (`running` → `run`, `doctors` → `doctor`)

**Amharic:**
- Ethiopic character normalization (variant forms mapped to canonical forms)
- Ethiopic punctuation removal
- Custom stopword list (common Amharic function words)
- Simple suffix stripping (common Amharic morphological suffixes)

### 2. Inverted Index

Structure:
```json
{
  "term": {
    "doc_id_1": 3,
    "doc_id_2": 1
  }
}
```

Additional structures:
- `df[term]` — document frequency (number of docs containing the term)
- `doc_lengths[doc_id]` — total token count per document (for BM25 normalization)
- `avg_doc_length` — used in BM25's length normalization factor

The index is saved to `saved_indexes/main_index.json` on first build and loaded from disk on subsequent restarts.

### 3. Ranking

**TF-IDF:**
```
score(d, q) = Σ (1 + log(tf(t,d))) × log(N/df(t))
```
Log-normalized TF dampens the effect of very frequent terms. IDF rewards terms that discriminate between documents.

**BM25 (k1=1.5, b=0.75):**
```
score(d, q) = Σ IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avg|d|))
```
BM25 saturates TF (doubling term frequency doesn't double score) and normalizes by document length, typically outperforming TF-IDF.

### 4. Cross-Lingual Retrieval (CLIR)

When a query arrives:
1. Detect query language (English vs Amharic based on Unicode block)
2. Preprocess query in its native language
3. Translate query terms using a bilingual dictionary (English↔Amharic)
4. Preprocess translated terms in the target language
5. Score same-language documents with original tokens
6. Score cross-language documents with translated tokens
7. Normalize scores to [0,1] within each group, boost same-language by 20%
8. Merge and re-rank all results

Example:
- Query: `"health technology"` (English)
- Translated: `['ጤና', 'ጤንነት', 'ቴክኖሎጂ']` (Amharic)
- Both English health docs AND Amharic health docs are returned

### 5. Query Expansion

When enabled, the system adds synonyms from a hand-crafted dictionary:
- `doctor` → also searches for `physician`, `surgeon`, `clinician`
- `education` → also searches for `learning`, `schooling`, `training`

Applied to both the original tokens and the cross-lingual translated tokens.

---

## Evaluation

The evaluation dashboard tests 15 standardized queries with manually assigned relevance judgments.

| Metric    | Formula                              | Meaning                                    |
|-----------|--------------------------------------|--------------------------------------------|
| Precision | \|Ret ∩ Rel\| / \|Ret\|             | What fraction of results are relevant?     |
| Recall    | \|Ret ∩ Rel\| / \|Rel\|             | What fraction of relevant docs were found? |
| F1        | 2 × P × R / (P + R)                 | Harmonic mean of precision and recall      |
| P@5       | \|top-5 ∩ Rel\| / 5                 | Precision on the first 5 results           |
| MAP       | Mean average precision across queries| Considers rank position of each hit        |

Both TF-IDF and BM25 are evaluated and compared side-by-side.

---

## Dataset

26 documents total:
- **13 English** documents covering: Health, Technology, Education, Environment, Sports, Agriculture, Economy, Science, Culture, Politics, Public Health, Digital Technology, Food & Nutrition
- **13 Amharic** documents on the same topics, allowing cross-lingual evaluation

Documents follow the format:
```
Title (line 1)
Category (line 2)
Body text (remaining lines)
```

---

## Future Improvements

- Vector space model with word embeddings (Word2Vec, fastText) for semantic search
- Expanded bilingual lexicon using automated alignment from parallel corpora
- Neural ranking (BM25 + BERT re-ranking)
- User relevance feedback (Rocchio algorithm)
- Named entity recognition for Amharic text
- Phrase search and Boolean operators in the UI
- Per-user search history stored in a database

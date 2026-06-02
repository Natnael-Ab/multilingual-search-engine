# Multilingual Search Engine

A multilingual Information Retrieval (IR) system that enables English and Amharic document retrieval using modern search techniques including BM25 ranking, TF-IDF ranking, query expansion, and cross-language search.

## Overview

This project demonstrates the core concepts of Information Retrieval by implementing a complete search engine capable of retrieving relevant documents across multiple languages. The system includes document preprocessing, indexing, ranking, evaluation, and a web-based search interface.

## Key Features

- English document retrieval
- Amharic document retrieval
- Cross-language information retrieval
- BM25 ranking algorithm
- TF-IDF ranking algorithm
- Query expansion
- Inverted index generation
- Search evaluation metrics
- Flask-based web interface
- Persistent index storage

## System Architecture

```text
User Query
     │
     ▼
Preprocessing
     │
     ▼
Query Expansion
     │
     ▼
Translation Layer
     │
     ▼
BM25 / TF-IDF Ranking
     │
     ▼
Search Results
```

## Project Structure

```text
multilingual-search-engine/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── dataset/
│   ├── english/
│   └── amharic/
│
├── ir/
│   ├── preprocessing.py
│   ├── indexing.py
│   ├── ranking.py
│   ├── search_engine.py
│   ├── translation.py
│   ├── query_expansion.py
│   └── evaluation.py
│
├── templates/
├── static/
├── saved_indexes/
│
├── .github/
├── .devcontainer/
│
└── render.yaml
```

## Technologies Used

- Python 3
- Flask
- Flask-CORS
- NLTK
- BM25 Ranking
- TF-IDF Ranking
- HTML
- CSS
- JavaScript

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/multilingual-search-engine.git
cd multilingual-search-engine
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK Resources

```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
```

### Run Application

```bash
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5000
```

## Search Features

The system supports:

- Keyword Search
- BM25 Ranking
- TF-IDF Ranking
- Query Expansion
- English Queries
- Amharic Queries
- Cross-Language Retrieval

## Evaluation Metrics

The retrieval system includes support for:

- Precision
- Recall
- F1 Score
- Precision@K
- Average Precision
- Mean Average Precision (MAP)

## Future Improvements

- Phrase Search
- Positional Indexing
- Spell Correction
- Semantic Search
- FastText Embeddings
- Transformer-Based Retrieval
- Hybrid BM25 + Vector Search

## Deployment

The project includes:

- GitHub Codespaces configuration
- GitHub Actions CI workflow
- Render deployment configuration
- Gunicorn production server setup

## License

This project is licensed under the MIT License.

## Author

- Natnael Abebe
- Gebeyehu Atanaw

Built as a multilingual Information Retrieval and Search Engine project using Python and Flask.
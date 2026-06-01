"""
Text preprocessing for English and Amharic documents.

We handle the two languages very differently because they have
different scripts and linguistic structures. English uses the
Porter stemmer; Amharic uses simple suffix stripping since no
standard stemmer exists for it yet.
"""

import re
import unicodedata
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data if it's not already cached
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Set up the Porter stemmer for English
_stemmer = PorterStemmer()

# English stopwords from NLTK — common words that don't carry meaning
_english_stopwords = set(stopwords.words('english'))

# Amharic stopwords — handcrafted list of common filler words in Amharic
# These words appear in almost every sentence so they don't help retrieval
_amharic_stopwords = {
    'እና', 'ነው', 'ናቸው', 'ያለ', 'ወይም', 'ከ', 'በ', 'ለ', 'እስከ', 'ወደ',
    'ላይ', 'ውስጥ', 'ስለ', 'እንደ', 'ግን', 'ስለዚህ', 'ነገር', 'ሆኖ', 'ሲሆን',
    'አለ', 'አልነበረም', 'ሆነ', 'ይህ', 'ይህን', 'ያ', 'ያን', 'እነዚህ', 'አንድ',
    'ሁሉ', 'ሌሎች', 'ዚህ', 'ሆነዋል', 'ተደርጓል', 'ኢትዮጵያ', 'ዓ', 'ም', 'ዓ.ም'
}

# Common Amharic suffixes — we strip these to reduce words to their root form
# This is a simple approach that works well enough for an IR system
_amharic_suffixes = [
    'ዎች', 'ዎቹ', 'ዎቼ', 'ዎቹን', 'ውን', 'ነው', 'ናቸው', 'ውም',
    'ችን', 'ቸው', 'ዋል', 'ዋቸው', 'ዋቸውን', 'ው', 'ን', 'ም', 'ና'
]


def detect_language(text: str) -> str:
    """
    Detect whether a piece of text is English or Amharic.

    We look at the Unicode blocks — Amharic (Ethiopic) characters
    fall in the range U+1200 to U+137F, so if most characters are
    there it's Amharic; otherwise we assume English.
    """
    amharic_count = 0
    latin_count = 0

    for char in text:
        code = ord(char)
        if 0x1200 <= code <= 0x137F:  # Ethiopic block
            amharic_count += 1
        elif 'a' <= char.lower() <= 'z':
            latin_count += 1

    if amharic_count > latin_count:
        return 'amharic'
    return 'english'


def normalize_english(text: str) -> list:
    """
    Clean and tokenize English text for indexing or querying.

    Steps:
      1. Lowercase everything so 'Health' and 'health' match
      2. Remove punctuation — it messes up token boundaries
      3. Split into words
      4. Remove stopwords (the, is, at, ...)
      5. Apply Porter stemming (running → run, doctors → doctor)

    Returns a list of normalized tokens.
    """
    # Step 1 — lowercase
    text = text.lower()

    # Step 2 — remove punctuation and non-alphabetic chars
    # We keep spaces so words stay separate
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Step 3 — split into tokens
    tokens = text.split()

    # Step 4 — remove stopwords and very short words (likely noise)
    tokens = [t for t in tokens if t not in _english_stopwords and len(t) > 1]

    # Step 5 — stem each token to its root form
    tokens = [_stemmer.stem(t) for t in tokens]

    return tokens


def normalize_amharic(text: str) -> list:
    """
    Clean and tokenize Amharic text for indexing or querying.

    Amharic uses the Ge'ez script (Ethiopic). We can't use an
    off-the-shelf stemmer, so we do lighter processing:
      1. Normalize visually similar characters (Ethiopic has variant forms)
      2. Remove punctuation (including Ethiopic punctuation marks)
      3. Split into words
      4. Remove stopwords
      5. Strip common suffixes to get approximate roots

    Returns a list of normalized tokens.
    """
    # Step 1 — normalize similar-looking Amharic characters
    # Some letters in Ethiopic have older and newer forms that look identical
    text = _normalize_amharic_chars(text)

    # Step 2 — remove punctuation (both ASCII and Ethiopic punctuation ። ፤ ፣)
    text = re.sub(r'[^\u1200-\u137F\s]', ' ', text)

    # Step 3 — split into tokens
    tokens = text.split()

    # Step 4 — remove stopwords
    tokens = [t for t in tokens if t not in _amharic_stopwords and len(t) > 1]

    # Step 5 — strip common suffixes to reduce words to roots
    tokens = [_strip_amharic_suffix(t) for t in tokens]

    # Remove any tokens that became too short after stripping
    tokens = [t for t in tokens if len(t) > 1]

    return tokens


def _normalize_amharic_chars(text: str) -> str:
    """
    Map visually equivalent Ethiopic characters to a single canonical form.

    Some letters in Amharic have older variants that look the same.
    Unifying them helps with matching — 'ሀ' and 'ሃ' mean the same sound.
    """
    # Map older/variant forms to their modern equivalents
    char_map = {
        'ሀ': 'ሃ', 'ሁ': 'ሁ', 'ሂ': 'ሂ', 'ሄ': 'ሄ', 'ሆ': 'ሆ',
        'ሐ': 'ሀ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሃ', 'ሔ': 'ሄ', 'ሕ': 'ሀ', 'ሖ': 'ሆ',
        'ኀ': 'ሀ', 'ኁ': 'ሁ', 'ኂ': 'ሂ', 'ኃ': 'ሃ', 'ኄ': 'ሄ', 'ኅ': 'ሀ', 'ኆ': 'ሆ',
        'ሰ': 'ሠ', 'ሱ': 'ሡ', 'ሲ': 'ሢ', 'ሳ': 'ሣ', 'ሴ': 'ሤ', 'ሶ': 'ሦ',
        'አ': 'ዐ', 'ኡ': 'ዑ', 'ኢ': 'ዒ', 'ኤ': 'ዔ', 'ኦ': 'ዖ',
    }
    for old, new in char_map.items():
        text = text.replace(old, new)
    return text


def _strip_amharic_suffix(word: str) -> str:
    """
    Remove common Amharic suffixes to approximate word roots.

    This is a greedy approach — we try longer suffixes first.
    We only strip if the remaining root is at least 2 characters.
    """
    # Sort by length descending — strip the longest match first
    for suffix in sorted(_amharic_suffixes, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[:-len(suffix)]
    return word


def preprocess(text: str, language: str = None) -> list:
    """
    Main entry point — preprocess text in the detected or specified language.

    If language is not given, we auto-detect it from the content.
    Returns a list of normalized tokens ready for indexing or ranking.
    """
    if language is None:
        language = detect_language(text)

    if language == 'amharic':
        return normalize_amharic(text)
    else:
        return normalize_english(text)

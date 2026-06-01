"""
Bilingual English-Amharic dictionary for cross-lingual retrieval.

The key idea: when a user types an English query, we also search
Amharic documents by translating the query terms into Amharic,
and vice versa. This is called Cross-Language Information Retrieval (CLIR).

We use a hand-crafted bilingual dictionary because it's simple,
explainable, and works well for a domain-specific IR system.
"""


# ── Core bilingual dictionary ─────────────────────────────────────────────────
# Each entry maps an English term to a list of Amharic equivalents (and back).
# We chose common academic and general-knowledge terms that appear in our corpus.

ENGLISH_TO_AMHARIC = {
    # Health and medicine
    'health':       ['ጤና', 'ጤንነት'],
    'medical':      ['ሕክምና', 'ህክምና'],
    'doctor':       ['ሐኪም', 'ዶክተር'],
    'hospital':     ['ሆስፒታል', 'ሕክምና ቤት'],
    'disease':      ['ሕመም', 'ህመም', 'ደዌ'],
    'medicine':     ['መድሐኒት', 'ሕክምና'],
    'patient':      ['ታካሚ', 'ሕሙም'],
    'vaccine':      ['ክትባት', 'ቫክሲን'],
    'treatment':    ['ህክምና', 'ሕክምና'],
    'surgery':      ['ቀዶ ሕክምና'],
    'nurse':        ['ነርስ', 'ሕሙምን ተንከባካቢ'],
    'epidemic':     ['ወረርሽኝ'],
    'nutrition':    ['ምግባዊ ጤና', 'አመጋገብ'],
    'diet':         ['አመጋገብ', 'ምግብ'],

    # Technology
    'technology':   ['ቴክኖሎጂ'],
    'computer':     ['ኮምፒዩተር'],
    'internet':     ['ኢንተርኔት'],
    'digital':      ['ዲጂታል'],
    'software':     ['ሶፍትዌር', 'ፕሮግራም'],
    'data':         ['ዳታ', 'መረጃ'],
    'network':      ['አውታረ መረብ', 'ኔትወርክ'],
    'artificial':   ['አርቲፊሻል', 'ሰው ሰራሽ'],
    'intelligence': ['አስተውሎ', 'ብልሃት'],
    'mobile':       ['ሞባይል', 'ተንቀሳቃሽ'],
    'programming':  ['ፕሮግራሚንግ', 'ኮዲንግ'],
    'innovation':   ['ፈጠራ', 'ኢኖቬሽን'],
    'electricity':  ['ኤሌክትሪክ'],
    'energy':       ['ኢነርጂ', 'ሃይል'],

    # Education
    'education':    ['ትምህርት'],
    'school':       ['ትምህርት ቤት', 'ትምህርትቤት'],
    'university':   ['ዩኒቨርሲቲ', 'ዩኒቨርስቲ'],
    'student':      ['ተማሪ', 'ተማሪዎች'],
    'teacher':      ['መምህር', 'አስተማሪ'],
    'learning':     ['ትምህርት', 'መማር'],
    'knowledge':    ['እውቀት', 'ዕውቀት'],
    'research':     ['ምርምር', 'ጥናት'],
    'study':        ['ጥናት', 'ትምህርት'],
    'academic':     ['አካዳሚያዊ', 'ትምህርታዊ'],
    'science':      ['ሳይንስ', 'ሣይንስ'],
    'mathematics':  ['ሒሳብ', 'ሂሳብ'],

    # Environment
    'environment':  ['አካባቢ'],
    'climate':      ['የአየር ሁኔታ', 'ክሊሜት'],
    'water':        ['ውሃ', 'ውሀ'],
    'forest':       ['ጫካ', 'ደን'],
    'pollution':    ['ብክለት', 'አካባቢ ብክለት'],
    'nature':       ['ተፈጥሮ'],
    'conservation': ['ጥበቃ', 'ጥምቀት'],
    'soil':         ['አፈር'],
    'agriculture':  ['ግብርና', 'እርሻ'],
    'farming':      ['እርሻ', 'ግብርና'],
    'farmer':       ['አርሶ አደር', 'ገበሬ'],
    'crop':         ['ሰብል', 'እህል'],
    'food':         ['ምግብ'],
    'animal':       ['እንስሳ', 'ከብት'],

    # Economy
    'economy':      ['ኢኮኖሚ'],
    'economic':     ['ኢኮኖሚያዊ'],
    'market':       ['ገበያ'],
    'trade':        ['ንግድ', 'ገበያ'],
    'business':     ['ንግድ', 'ቢዝነስ'],
    'investment':   ['ኢንቨስትመንት', 'ልማት'],
    'bank':         ['ባንክ'],
    'money':        ['ገንዘብ'],
    'price':        ['ዋጋ'],
    'industry':     ['ኢንዱስትሪ', 'ኢንቨስትመንት'],
    'development':  ['ልማት', 'እድገት'],
    'growth':       ['እድገት', 'ዕድገት'],

    # Culture and society
    'culture':      ['ባህል'],
    'tradition':    ['ባህል', 'ወግ'],
    'art':          ['ጥበብ', 'ኪነጥበብ'],
    'music':        ['ሙዚቃ'],
    'literature':   ['ስነ ጽሁፍ', 'ሥነ-ጽሑፍ'],
    'language':     ['ቋንቋ'],
    'history':      ['ታሪክ'],
    'religion':     ['ሃይማኖት', 'ሀይማኖት'],
    'community':    ['ማህበረሰብ', 'ማህበረስብ'],
    'society':      ['ማህበረሰብ'],
    'family':       ['ቤተሰብ'],
    'women':        ['ሴቶች', 'ሴት'],
    'children':     ['ልጆች', 'ህጻናት'],

    # Politics / governance
    'government':   ['መንግሥት', 'መንግስት'],
    'politics':     ['ፖለቲካ'],
    'democracy':    ['ዲሞክራሲ'],
    'election':     ['ምርጫ'],
    'law':          ['ህግ', 'ሕግ'],
    'rights':       ['መብት'],
    'peace':        ['ሰላም'],
    'war':          ['ጦርነት'],
    'leader':       ['መሪ', 'ሊደር'],
    'policy':       ['ፖሊሲ', 'ህግ'],

    # Sport
    'sport':        ['ስፖርት'],
    'sports':       ['ስፖርት'],
    'football':     ['እግር ኳስ', 'ፊትቦል'],
    'athlete':      ['አትሌት'],
    'competition':  ['ውድድር', 'ፉክክር'],
    'training':     ['ስልጠና'],
    'team':         ['ቡድን'],
    'running':      ['ሩጫ', 'መሮጥ'],
    'champion':     ['ሻምፒዮን'],
    'olympic':      ['ኦሎምፒክ'],
}

# Build the reverse dictionary (Amharic → English) automatically
AMHARIC_TO_ENGLISH = {}
for eng_term, amharic_list in ENGLISH_TO_AMHARIC.items():
    for am_term in amharic_list:
        if am_term not in AMHARIC_TO_ENGLISH:
            AMHARIC_TO_ENGLISH[am_term] = []
        if eng_term not in AMHARIC_TO_ENGLISH[am_term]:
            AMHARIC_TO_ENGLISH[am_term].append(eng_term)


def translate_terms(terms: list, source_lang: str) -> list:
    """
    Translate a list of query terms from one language to the other.

    For cross-lingual retrieval we expand the query: if the user
    typed English, we also generate Amharic equivalents so we can
    search Amharic documents. And vice versa.

    Returns a flat list of translated terms (may contain duplicates removed).
    """
    translated = []

    if source_lang == 'english':
        # Translate English terms → Amharic
        for term in terms:
            amharic_equivalents = ENGLISH_TO_AMHARIC.get(term, [])
            translated.extend(amharic_equivalents)

            # Also try stemmed/partial matches — helpful for word-form variations
            for key, values in ENGLISH_TO_AMHARIC.items():
                if term in key or key in term:
                    translated.extend(values)

    elif source_lang == 'amharic':
        # Translate Amharic terms → English
        for term in terms:
            english_equivalents = AMHARIC_TO_ENGLISH.get(term, [])
            translated.extend(english_equivalents)

            # Also try partial key matches
            for key, values in AMHARIC_TO_ENGLISH.items():
                if term in key or key in term:
                    translated.extend(values)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in translated:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique


def get_cross_lingual_terms(raw_query: str, source_lang: str, preprocessor) -> list:
    """
    Given a raw query string and its language, return preprocessed tokens
    in the OPPOSITE language for cross-lingual document retrieval.

    This is the main function called by the search engine to enable
    searching across language boundaries.
    """
    # First, preprocess the original query tokens
    original_tokens = preprocessor(raw_query, source_lang)

    # Translate to the other language
    translated = translate_terms(original_tokens, source_lang)

    # Preprocess the translated terms in the target language
    target_lang = 'amharic' if source_lang == 'english' else 'english'
    target_tokens = []
    for term in translated:
        target_tokens.extend(preprocessor(term, target_lang))

    return list(set(target_tokens))

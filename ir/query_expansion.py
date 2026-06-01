"""
Query expansion using a synonym dictionary.

Query expansion improves recall by adding related terms to the
user's query. For example, if someone searches for 'doctor', we
also search for 'physician', 'surgeon', etc.

We use a simple hand-crafted synonym dictionary — in a production
system you'd use WordNet or a word embedding model, but for this
university project the dictionary approach is transparent and
easy to explain in a viva.
"""


# Synonym dictionary for English
# Each key is a term, and its value is a list of synonyms to add
ENGLISH_SYNONYMS = {
    # Health
    'health':       ['medical', 'wellness', 'wellbeing', 'healthcare'],
    'doctor':       ['physician', 'surgeon', 'clinician', 'practitioner'],
    'hospital':     ['clinic', 'medical center', 'infirmary', 'healthcare facility'],
    'disease':      ['illness', 'sickness', 'condition', 'disorder', 'ailment'],
    'medicine':     ['drug', 'medication', 'remedy', 'pharmaceutical', 'treatment'],
    'patient':      ['sick person', 'individual', 'case'],
    'vaccine':      ['vaccination', 'immunization', 'jab', 'shot'],
    'nutrition':    ['diet', 'eating', 'food intake', 'nourishment'],
    'epidemic':     ['outbreak', 'pandemic', 'spread', 'infection'],

    # Technology
    'technology':   ['tech', 'innovation', 'digital', 'computing'],
    'computer':     ['machine', 'system', 'device', 'pc'],
    'internet':     ['web', 'online', 'network', 'connectivity'],
    'software':     ['program', 'application', 'app', 'code'],
    'data':         ['information', 'records', 'dataset', 'statistics'],
    'artificial':   ['synthetic', 'automated', 'machine'],
    'intelligence': ['smart', 'cognitive', 'learning', 'reasoning'],
    'mobile':       ['phone', 'smartphone', 'device', 'portable'],
    'programming':  ['coding', 'software development', 'scripting'],

    # Education
    'education':    ['learning', 'schooling', 'training', 'teaching'],
    'school':       ['institution', 'college', 'academy', 'education'],
    'university':   ['college', 'institute', 'higher education', 'campus'],
    'student':      ['learner', 'pupil', 'scholar', 'trainee'],
    'teacher':      ['educator', 'instructor', 'professor', 'tutor'],
    'research':     ['study', 'investigation', 'analysis', 'inquiry'],
    'knowledge':    ['understanding', 'expertise', 'information', 'wisdom'],
    'science':      ['scientific', 'biology', 'chemistry', 'physics', 'research'],
    'learning':     ['education', 'study', 'training', 'skill'],

    # Environment
    'environment':  ['nature', 'ecology', 'habitat', 'surroundings'],
    'climate':      ['weather', 'temperature', 'atmosphere', 'global'],
    'water':        ['river', 'lake', 'rainfall', 'irrigation'],
    'forest':       ['woodland', 'trees', 'vegetation', 'jungle'],
    'pollution':    ['contamination', 'waste', 'emission', 'toxic'],
    'conservation': ['protection', 'preservation', 'sustainability'],
    'agriculture':  ['farming', 'cultivation', 'crops', 'harvest'],
    'farmer':       ['grower', 'cultivator', 'agriculturist'],
    'food':         ['nutrition', 'diet', 'meal', 'eating', 'harvest'],

    # Economy
    'economy':      ['economic', 'finance', 'trade', 'business', 'market'],
    'market':       ['trade', 'commerce', 'exchange', 'business'],
    'business':     ['enterprise', 'company', 'commerce', 'trade'],
    'trade':        ['commerce', 'exchange', 'export', 'import'],
    'investment':   ['funding', 'capital', 'finance', 'development'],
    'development':  ['growth', 'progress', 'advancement', 'improvement'],

    # Culture
    'culture':      ['tradition', 'heritage', 'customs', 'arts'],
    'art':          ['music', 'painting', 'literature', 'creativity'],
    'music':        ['song', 'melody', 'rhythm', 'instrument'],
    'history':      ['past', 'heritage', 'civilization', 'tradition'],
    'community':    ['society', 'people', 'group', 'neighborhood'],
    'language':     ['tongue', 'dialect', 'communication', 'speech'],

    # Politics
    'government':   ['state', 'authority', 'administration', 'regime'],
    'democracy':    ['election', 'vote', 'rights', 'freedom'],
    'election':     ['vote', 'poll', 'ballot', 'campaign'],
    'law':          ['regulation', 'rule', 'legislation', 'justice'],
    'rights':       ['freedom', 'liberty', 'justice', 'equality'],
    'peace':        ['stability', 'harmony', 'security', 'agreement'],

    # Sports
    'sport':        ['athletic', 'game', 'competition', 'exercise', 'fitness'],
    'football':     ['soccer', 'sport', 'game', 'match'],
    'athlete':      ['player', 'sportsperson', 'runner', 'champion'],
    'competition':  ['tournament', 'contest', 'race', 'match'],
    'training':     ['practice', 'exercise', 'preparation', 'coaching'],
    'champion':     ['winner', 'victor', 'title holder', 'gold medalist'],
}

# Lightweight Amharic synonym set
# In Amharic we keep this small and conservative to avoid noise
AMHARIC_SYNONYMS = {
    'ጤና':      ['ጤንነት', 'ጥሩ ጤና'],
    'ሕክምና':    ['ህክምና', 'ሕክምና'],
    'ትምህርት':   ['ትምህርት ቤት', 'ትምህርታዊ', 'ቤት ትምህርት'],
    'ቴክኖሎጂ':   ['ቴክ', 'ዘዴ'],
    'ኢኮኖሚ':    ['ኢኮኖሚያዊ', 'ንግድ'],
    'ግብርና':    ['እርሻ', 'አርሶ'],
    'ባህል':     ['ወግ', 'ልማድ'],
    'ሳይንስ':    ['ጥናት', 'ምርምር'],
    'ስፖርት':    ['ውድድር', 'ልምምድ'],
    'አካባቢ':    ['ተፈጥሮ', 'ጥበቃ'],
    'ምርምር':    ['ጥናት', 'ምርምር ስራ'],
    'ህመም':     ['ሕመም', 'ደዌ'],
}


def expand_query(tokens: list, language: str) -> list:
    """
    Expand a list of preprocessed tokens by adding synonyms.

    This is optional — users can toggle it in the UI. When enabled,
    it improves recall by catching documents that use different words
    for the same concept.

    Returns the original tokens + new synonym tokens (deduplicated).
    """
    expanded = list(tokens)  # start with original tokens

    if language == 'english':
        synonym_dict = ENGLISH_SYNONYMS
    else:
        synonym_dict = AMHARIC_SYNONYMS

    for token in tokens:
        synonyms = synonym_dict.get(token, [])
        for synonym in synonyms:
            # Synonyms can be multi-word — split and add each word
            parts = synonym.split()
            expanded.extend(parts)

    # Remove duplicates while preserving original order
    seen = set()
    unique = []
    for t in expanded:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique


def get_suggestions(partial_query: str, language: str) -> list:
    """
    Return autocomplete suggestions based on the partial query.

    We look up all known terms (dictionary keys + synonyms) that
    start with the partial query string. This powers the search
    bar's autocomplete dropdown.
    """
    partial = partial_query.lower().strip()
    if not partial:
        return []

    suggestions = []

    if language == 'english':
        all_terms = list(ENGLISH_SYNONYMS.keys())
        # Also add some common multi-word suggestions
        common_phrases = [
            'health technology', 'artificial intelligence', 'climate change',
            'food security', 'public health', 'sustainable development',
            'higher education', 'digital transformation', 'human rights',
            'economic growth', 'environmental conservation', 'disease prevention',
            'community development', 'scientific research', 'political democracy',
        ]
        all_terms.extend(common_phrases)
        suggestions = [t for t in all_terms if t.startswith(partial)]
    else:
        all_terms = list(AMHARIC_SYNONYMS.keys())
        suggestions = [t for t in all_terms if t.startswith(partial)]

    return suggestions[:8]  # return at most 8 suggestions

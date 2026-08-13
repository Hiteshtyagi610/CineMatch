"""
Shared text-preprocessing helpers.

Both `scripts/train_model.py` (offline training) and `backend/main.py`
(online inference, for the /recommend/vibe free-text endpoint) import this
module instead of each keeping their own copy. That's the whole point: a
query typed at request time has to be cleaned exactly the same way the
training corpus was cleaned, or it lands in a different corner of the TF-IDF
space and similarity scores stop meaning anything. Keeping this logic in one
place is what makes that guarantee possible instead of just "true today."
"""
from __future__ import annotations

import ast
import functools
import re

_TOKEN_RE = re.compile(r"[^a-z\s]")


@functools.lru_cache(maxsize=1)
def _load_nltk_assets():
    """Lazily load stopwords + lemmatizer, cached after first call.

    Falls back to sklearn's built-in stopword list (no lemmatization) if
    NLTK corpora can't be downloaded — e.g. a sandboxed/offline deploy
    target — so the pipeline degrades gracefully instead of crashing.
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        for pkg in ("stopwords", "wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)

        return set(stopwords.words("english")), WordNetLemmatizer()
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return set(ENGLISH_STOP_WORDS), None


def clean_text(text: str) -> str:
    """Lowercase, strip non-alphabetic characters, drop stopwords/short
    tokens, lemmatize. Mirrors the cleaning applied to the training corpus
    in scripts/train_model.py exactly."""
    stop_words, lemmatizer = _load_nltk_assets()

    text = str(text).lower()
    text = _TOKEN_RE.sub(" ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def parse_genres(raw: str) -> str:
    """TMDB stores genres as a stringified list of dicts -> ' '.join(names)."""
    try:
        return " ".join(g["name"] for g in ast.literal_eval(raw))
    except (ValueError, SyntaxError):
        return ""

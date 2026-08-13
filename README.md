# 🎬 CineMatch — Content-Based Movie Recommender

A movie recommendation system that combines an **NLP content-similarity model
(TF-IDF + cosine similarity, popularity-aware re-ranking, free-text "vibe"
search)** trained on 45k+ movies with **live TMDB data**, served through a
FastAPI backend and a Streamlit UI.

Search a movie → get plot/theme-similar recommendations from the local NLP
model, plus a genre-matched feed pulled live from TMDB. Or skip the title
entirely and describe what you want to watch — the same TF-IDF space powers
free-text search too.

## Tech stack

| Layer          | Tools |
|----------------|-------|
| NLP / modeling | scikit-learn (TF-IDF, cosine similarity), NLTK (stopwords, lemmatization), pandas, NumPy |
| Backend API    | FastAPI, httpx (async TMDB calls), Pydantic |
| Frontend       | Streamlit, custom CSS |
| Data source    | [TMDB](https://www.themoviedb.org/) movie metadata (45k+ titles) + live TMDB API |

## Architecture

```
data/movies_metadata.csv
        │
        ▼
scripts/train_model.py ──► models/*.pkl  (TF-IDF matrix, vectorizer, cleaned df, title index)
        │        │
        │        ▼
        │   common/preprocess.py   ◄── imported by BOTH the training script and the API,
        │   (clean_text, parse_genres)  so a query typed at request time is cleaned exactly
        │        ▲                      the same way the training corpus was — no drift.
        │        │
notebooks/nlp_pipeline.ipynb    backend/main.py  (FastAPI)
  (EDA + walkthrough,                 │  - /movie/search    → details + TF-IDF recs + genre recs
   same pipeline, documented)         │  - /recommend/tfidf → title-based recs (popularity-blended)
                                      │  - /recommend/vibe  → free-text recs (same TF-IDF space)
                                      │  - /tmdb/search      → live TMDB search
                                      │  - /home             → trending/popular/etc.
                                      ▼
                          frontend/app.py  (Streamlit UI)
```

## How the recommendation model works

1. **Merge text signal** per movie: `overview + genres (2x weighted) + tagline`
2. **Clean**: lowercase, strip non-alphabetic characters, drop stopwords,
   lemmatize (NLTK, with an automatic fallback to scikit-learn's stopword
   list if NLTK corpora can't be downloaded in the current environment)
3. **Vectorize**: `TfidfVectorizer(max_features=50000, ngram_range=(1,2))` —
   unigrams and bigrams
4. **Rank**: cosine similarity between the query vector and every movie via
   a normalized dot product (TF-IDF vectors are already L2-normalized), with
   `np.argpartition` for O(n) top-k selection instead of a full sort
5. **Re-rank**: blend content similarity with a normalized
   popularity/vote-count signal (`alpha` parameter, default `0.85`) so
   results are similar *and* worth watching, not just lexically close
6. **Free-text search**: the same fitted vectorizer transforms an arbitrary
   query string ("a time-loop heist where the crew doesn't trust each
   other") into the same vector space — `/recommend/vibe` needs no separate
   embedding model

The full walkthrough with EDA and inline explanation is in
[`notebooks/nlp_pipeline.ipynb`](notebooks/nlp_pipeline.ipynb).

## Project structure

```
.
├── data/
│   └── movies_metadata.csv       # raw TMDB metadata dump (gitignored, see Setup)
├── common/
│   └── preprocess.py             # clean_text / parse_genres — shared by script + API
├── notebooks/
│   └── nlp_pipeline.ipynb        # documented EDA + model walkthrough
├── scripts/
│   └── train_model.py            # production training script
├── models/                       # generated artifacts (gitignored, regenerate locally)
│   ├── df.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   └── title_indices.pkl
├── backend/
│   └── main.py                   # FastAPI app
├── frontend/
│   └── app.py                    # Streamlit app
├── requirements.txt
├── Dockerfile                    # backend container
├── .env.example
└── runtime.txt
```

## Setup

```bash
git clone https://github.com/Hiteshtyagi610/CineMatch
cd cinematch
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set TMDB_API_KEY (free key: https://www.themoviedb.org/settings/api)
```

### 1. Train the model (only needed once, or after changing the data)

```bash
python scripts/train_model.py
```

This regenerates everything in `models/` from `data/movies_metadata.csv`.

### 2. Run the API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 3. Run the UI

In a second terminal:

```bash
cd frontend
streamlit run app.py
```

By default the frontend calls `http://127.0.0.1:8000`. To point it at a
deployed API instead, set `API_BASE`:

```bash
API_BASE=https://your-deployed-api.onrender.com streamlit run frontend/app.py
```

## API reference

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /home?category=trending\|popular\|top_rated\|now_playing\|upcoming&limit=` | Home feed cards |
| `GET /tmdb/search?query=` | Live TMDB title search |
| `GET /movie/id/{tmdb_id}` | Full movie details |
| `GET /movie/search?query=&alpha=` | Details + TF-IDF recs + genre recs bundle |
| `GET /recommend/tfidf?title=&top_n=&alpha=` | Title-based recs, popularity-blended (debug/inspection) |
| `GET /recommend/vibe?query=&top_n=` | Free-text "describe a vibe" recs |
| `GET /recommend/genre?tmdb_id=&limit=` | Genre-based recommendations (TMDB discover) |

`alpha` (0–1, default 0.85) controls how much the ranking leans on content
similarity vs. a popularity/vote-count "quality" signal — `1.0` is pure
content-based (what the original prototype always did).

## Notes on fixes made to the original pipeline

- **Lemmatization bug**: the original preprocessing computed lemmatized
  tokens into a variable that was never used, so lemmatization silently
  never ran. Fixed in `common/preprocess.py`.
- **Duplicate titles**: some titles appear more than once in the raw dataset
  (remakes, re-releases); the title→index lookup now keeps the first
  occurrence so lookups always resolve to a single row instead of raising
  ambiguous-truth-value errors.
- **File path mismatch**: the API previously looked for `tfidf_matrix.pkl`
  while the notebook saved `matrix.pkl` — the model would fail to load on a
  fresh clone. Naming is now consistent across the training script and API.
- **Exposed API key**: an earlier draft of this project had a live TMDB key
  committed in `.env`. `.env` is gitignored and `.env.example` ships with a
  placeholder — **rotate any key that was ever committed, before pushing
  this anywhere public.**
- **Preprocessing drift**: the training script and API used to keep their
  own separate copies of the text-cleaning logic — a classic way for a
  notebook and a production service to quietly diverge. Both now import
  `clean_text`/`parse_genres` from `common/preprocess.py`, so a `/recommend/vibe`
  query is guaranteed to be cleaned identically to the training corpus.
- **No offline fallback for NLTK**: if `nltk.download()` can't reach the
  internet (locked-down deploy target, CI sandbox), the pipeline now falls
  back to scikit-learn's built-in stopword list instead of crashing.

## Possible extensions

- Swap TF-IDF for sentence embeddings (e.g. `all-MiniLM-L6-v2`) for semantic
  (not just lexical) similarity
- Cache TMDB responses (Redis) to cut latency on repeat lookups
- User accounts + watchlist, with click-through tracked as implicit feedback
  to tune `alpha` per user instead of a global default
- A small eval harness (genre-overlap precision@k) as a proxy metric for
  recommendation quality when experimenting with the pipeline


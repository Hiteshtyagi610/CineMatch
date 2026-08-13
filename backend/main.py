import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from common.preprocess import clean_text  # noqa: E402 — same cleaning used at training time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise RuntimeError(
        "TMDB_API_KEY missing. Copy .env.example to .env and set your TMDB key."
    )

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

MODELS_DIR = BASE_DIR / "models"

DF_PATH = MODELS_DIR / "df.pkl"
INDICES_PATH = MODELS_DIR / "title_indices.pkl"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

app = FastAPI(title="Movie Recommender API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared HTTP client for TMDB calls
# ---------------------------------------------------------------------------
# One pooled/keep-alive client for the whole app lifetime, instead of opening
# a brand new TCP+TLS connection per request. This also matters on Windows
# setups where antivirus/network-inspection tools intermittently drop fresh
# outbound connections (shows up as random httpx.ConnectError / 502s) —
# reusing connections + retrying on failure papers over that instability.
http_client: httpx.AsyncClient = httpx.AsyncClient(
    timeout=httpx.Timeout(20.0, connect=10.0),
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
    transport=httpx.AsyncHTTPTransport(retries=3),
)

# ---------------------------------------------------------------------------
# In-memory model state (loaded once at startup)
# ---------------------------------------------------------------------------
df: Optional[pd.DataFrame] = None
tfidf_matrix: Any = None
tfidf_vectorizer: Any = None
title_to_idx: Dict[str, int] = {}
pop_norm: Optional[np.ndarray] = None
votes_norm: Optional[np.ndarray] = None


def _minmax(series: pd.Series) -> np.ndarray:
    arr = series.to_numpy(dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    vote_average: Optional[float] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


class VibeSearchResponse(BaseModel):
    query: str
    results: List[TFIDFRecItem]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    return f"{TMDB_IMG_500}{path}" if path else None


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Call the TMDB API using the shared, pooled client with retry-on-failure.

    Retries a few times on connection-level errors (timeouts, dropped
    connections, DNS blips) before giving up — a single flaky connection
    attempt no longer surfaces straight to the user as a 502.
    """
    if not TMDB_API_KEY:
        raise HTTPException(status_code=500, detail="TMDB_API_KEY not set")
    query = {**params, "api_key": TMDB_API_KEY}

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = await http_client.get(f"{TMDB_BASE}{path}", params=query)
        except httpx.RequestError as e:
            last_error = e
            continue  # retry

        if resp.status_code != 200:
            raise HTTPException(
                status_code=502, detail=f"TMDB error {resp.status_code}: {resp.text}"
            )
        return resp.json()

    raise HTTPException(
        status_code=502, detail=f"TMDB request error after 3 attempts: {last_error!r}"
    )


async def tmdb_cards_from_results(results: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
    out = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        vote_average=data.get("vote_average"),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie",
        {"query": query, "include_adult": "false", "language": "en-US", "page": page},
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# TF-IDF recommendation (local NLP model)
# ---------------------------------------------------------------------------
def get_local_idx_by_title(title: str) -> int:
    key = _norm_title(title)
    if key in title_to_idx:
        return title_to_idx[key]
    raise HTTPException(status_code=404, detail=f"Title not found in local dataset: '{title}'")


def _blend_and_rank(
    scores: np.ndarray, exclude_idx: Optional[int], top_n: int, alpha: float
) -> List[Tuple[str, float]]:
    """Popularity-aware re-ranking.

    Pure TF-IDF similarity can surface obscure titles that share a handful
    of rare words with the query but aren't good recommendations. We blend
    in a normalized popularity/vote-count "quality" signal so results are
    both similar AND worth watching. alpha=1.0 is pure content similarity
    (what the original prototype always did); lower alpha leans more on
    popularity. Selection still uses argpartition for O(n) top-k.
    """
    quality = 0.6 * pop_norm + 0.4 * votes_norm
    blended = alpha * scores + (1 - alpha) * quality

    k = min(top_n + 1, len(blended))
    top_idx = np.argpartition(-blended, k - 1)[:k]
    top_idx = top_idx[np.argsort(-blended[top_idx])]

    out: List[Tuple[str, float]] = []
    for i in top_idx:
        if exclude_idx is not None and int(i) == exclude_idx:
            continue
        out.append((str(df.iloc[int(i)]["title"]), float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10, alpha: float = 0.85
) -> List[Tuple[str, float]]:
    """Cosine similarity via normalized dot product; O(n) top-k selection,
    re-ranked with a popularity-aware blend (see _blend_and_rank)."""
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF model not loaded")

    idx = get_local_idx_by_title(query_title)
    query_vec = tfidf_matrix[idx]
    scores = (tfidf_matrix @ query_vec.T).toarray().ravel()
    return _blend_and_rank(scores, exclude_idx=idx, top_n=top_n, alpha=alpha)


def tfidf_recommend_by_text(query: str, top_n: int = 12) -> List[Tuple[str, float]]:
    """Free-text 'vibe' search: clean + vectorize arbitrary text with the
    same fitted vectorizer used at training time, then rank the corpus
    against it directly. No separate embedding model needed — everything
    already lives in one TF-IDF space. Pure content similarity (no
    popularity blend — there's no anchor movie to blend against)."""
    if df is None or tfidf_matrix is None or tfidf_vectorizer is None:
        raise HTTPException(status_code=500, detail="TF-IDF model not loaded")

    cleaned = clean_text(query)
    query_vec = tfidf_vectorizer.transform([cleaned])
    scores = (tfidf_matrix @ query_vec.T).toarray().ravel()
    return _blend_and_rank(scores, exclude_idx=None, top_n=top_n, alpha=1.0)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_model_artifacts():
    global df, tfidf_matrix, tfidf_vectorizer, title_to_idx, pop_norm, votes_norm

    for path in (DF_PATH, INDICES_PATH, TFIDF_MATRIX_PATH, TFIDF_VECTORIZER_PATH):
        if not path.exists():
            raise RuntimeError(
                f"Missing model artifact: {path}. Run `python scripts/train_model.py` first."
            )

    df = pd.read_pickle(DF_PATH)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(TFIDF_VECTORIZER_PATH, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    title_to_idx = {_norm_title(k): int(v) for k, v in indices_obj.items()}

    if "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a 'title' column")

    pop_norm = _minmax(df["popularity"]) if "popularity" in df.columns else np.zeros(len(df))
    votes_norm = _minmax(df["vote_count"]) if "vote_count" in df.columns else np.zeros(len(df))


@app.on_event("shutdown")
async def close_http_client():
    await http_client.aclose()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "movies_loaded": None if df is None else len(df)}


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(category: str = Query("popular"), limit: int = Query(24, ge=1, le=50)):
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
        elif category in {"popular", "top_rated", "upcoming", "now_playing"}:
            data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        else:
            raise HTTPException(status_code=400, detail="Invalid category")
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


@app.get("/tmdb/search")
async def tmdb_search(query: str = Query(..., min_length=1), page: int = Query(1, ge=1, le=10)):
    """Raw TMDB search shape — used for autocomplete + result grid."""
    return await tmdb_search_movies(query=query, page=page)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(tmdb_id: int = Query(...), limit: int = Query(18, ge=1, le=50)):
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []
    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1},
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
    alpha: float = Query(0.85, ge=0.0, le=1.0, description="1.0 = pure content similarity; lower leans on popularity"),
):
    recs = tfidf_recommend_titles(title, top_n=top_n, alpha=alpha)
    return [{"title": t, "score": s} for t, s in recs]


@app.get("/recommend/vibe", response_model=VibeSearchResponse)
async def recommend_vibe(query: str = Query(..., min_length=3), top_n: int = Query(12, ge=1, le=30)):
    """Free-text 'describe what you want to watch' search, e.g.
    'a time-loop heist where the crew doesn't trust each other'."""
    recs = tfidf_recommend_by_text(query, top_n=top_n)
    items: List[TFIDFRecItem] = []
    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        items.append(TFIDFRecItem(title=title, score=score, tmdb=card))
    return VibeSearchResponse(query=query, results=items)


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
    alpha: float = Query(0.85, ge=0.0, le=1.0),
):
    """Details + local TF-IDF recs + TMDB genre recs for the best TMDB match."""
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404, detail=f"No TMDB movie found for query: {query}")

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    tfidf_items: List[TFIDFRecItem] = []
    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n, alpha=alpha)
    except HTTPException:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n, alpha=alpha)
        except HTTPException:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1},
        )
        cards = await tmdb_cards_from_results(discover.get("results", []), limit=genre_limit)
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
"""
Streamlit frontend for the Movie Recommender.

Talks to the FastAPI backend (see backend/main.py) for everything —
TMDB search/home feeds and the local TF-IDF recommendation model.
This file owns presentation only: theme, layout, cards, routing.
"""

import os

import requests
import streamlit as st

# =============================================================================
# CONFIG
# =============================================================================
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
BACKDROP_IMG = "https://image.tmdb.org/t/p/original"
PLACEHOLDER_POSTER = "https://placehold.co/500x750/141414/e50914?text=No+Poster"

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# =============================================================================
# THEME
# =============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at 20% -10%, #241014 0%, #0a0a0c 45%) fixed;
    color: #f2f2f2;
}

/* hide default streamlit chrome for a cleaner app feel */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }

/* ---------- Hero ---------- */
.hero {
    position: relative;
    border-radius: 20px;
    overflow: hidden;
    padding: 48px 40px;
    margin-bottom: 28px;
    background: linear-gradient(100deg, rgba(10,10,12,0.96) 15%, rgba(10,10,12,0.55) 65%, rgba(10,10,12,0.15) 100%),
                var(--hero-img, linear-gradient(120deg,#1a1015,#0a0a0c));
    background-size: cover;
    background-position: center 20%;
    border: 1px solid rgba(255,255,255,0.06);
}
.hero-eyebrow {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #e50914;
    background: rgba(229,9,20,0.12);
    border: 1px solid rgba(229,9,20,0.35);
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #ffffff, #b8b8b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub { color: #b3b3b3; font-size: 1.02rem; max-width: 640px; line-height: 1.5; }

/* ---------- Section headers ---------- */
.section-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin: 30px 0 14px 0;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #fafafa;
    border-left: 4px solid #e50914;
    padding-left: 10px;
}
.section-tag {
    font-size: 0.78rem;
    color: #7a7a7a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ---------- Movie card ---------- */
.movie-card {
    border-radius: 14px;
    overflow: hidden;
    background: #141416;
    border: 1px solid rgba(255,255,255,0.06);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    margin-bottom: 10px;
}
.movie-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 14px 28px rgba(0,0,0,0.55);
    border-color: rgba(229,9,20,0.5);
}
.poster-wrap { position: relative; width: 100%; aspect-ratio: 2/3; overflow: hidden; background: #1c1c1e; }
.poster-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
.rating-badge {
    position: absolute; top: 8px; right: 8px;
    background: rgba(10,10,12,0.85);
    border: 1px solid rgba(255,255,255,0.15);
    color: #f5c518;
    font-size: 0.74rem; font-weight: 700;
    padding: 3px 7px; border-radius: 8px;
    backdrop-filter: blur(4px);
}
.card-body { padding: 10px 12px 12px 12px; }
.card-title {
    font-size: 0.86rem; font-weight: 600; color: #f2f2f2;
    line-height: 1.2rem; height: 2.4rem; overflow: hidden;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.card-year { font-size: 0.75rem; color: #8a8a8a; margin-top: 2px; }
.match-pill {
    display: inline-block; margin-top: 6px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.03em;
    color: #4ade80; background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.3);
    padding: 2px 8px; border-radius: 999px;
}

/* Streamlit buttons -> ghost "view" buttons under each card */
div[data-testid="stButton"] > button {
    width: 100%;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    color: #e6e6e6;
    border-radius: 8px;
    padding: 4px 0;
    font-size: 0.78rem;
    font-weight: 600;
    transition: all 0.15s ease;
}
div[data-testid="stButton"] > button:hover {
    background: #e50914;
    border-color: #e50914;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d0f;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Search input */
div[data-testid="stTextInput"] input {
    background: #17171a;
    border: 1px solid rgba(255,255,255,0.12);
    color: #f2f2f2;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 1rem;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #e50914;
    box-shadow: 0 0 0 1px #e50914;
}

/* Vibe search box */
div[data-testid="stTextArea"] textarea {
    background: #17171a;
    border: 1px solid rgba(255,255,255,0.12);
    color: #f2f2f2;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 0.95rem;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #e50914;
    box-shadow: 0 0 0 1px #e50914;
}
.vibe-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #7a7a7a;
    margin: 22px 0 8px 0;
}

/* Detail page info card */
.detail-card {
    background: #141416;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px 26px;
}
.genre-chip {
    display: inline-block;
    background: rgba(229,9,20,0.12);
    border: 1px solid rgba(229,9,20,0.35);
    color: #ff6b71;
    font-size: 0.75rem; font-weight: 600;
    padding: 3px 10px; border-radius: 999px;
    margin: 0 6px 6px 0;
}
.meta-row { color: #9a9a9a; font-size: 0.88rem; margin-bottom: 4px; }
hr { border-color: rgba(255,255,255,0.08); }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# STATE + ROUTING
# =============================================================================
if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except ValueError:
        pass


def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    st.query_params.pop("id", None)
    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()


# =============================================================================
# API HELPERS
# =============================================================================
@st.cache_data(ttl=30)
def api_get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"


def to_cards_from_tfidf_items(tfidf_items):
    cards = []
    for x in tfidf_items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            cards.append(
                {
                    "tmdb_id": tmdb["tmdb_id"],
                    "title": tmdb.get("title") or x.get("title") or "Untitled",
                    "poster_url": tmdb.get("poster_url"),
                    "vote_average": tmdb.get("vote_average"),
                    "release_date": tmdb.get("release_date"),
                    "match_pct": round((x.get("score") or 0) * 100),
                }
            )
    return cards

def parse_tmdb_search_to_cards(data, keyword: str, limit: int = 24):
    keyword_l = keyword.strip().lower()

    if isinstance(data, dict) and "results" in data:
        raw = data.get("results") or []
        raw_items = []
        for m in raw:
            title = (m.get("title") or "").strip()
            tmdb_id = m.get("id")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": f"{TMDB_IMG}{m['poster_path']}" if m.get("poster_path") else None,
                    "release_date": m.get("release_date", ""),
                    "vote_average": m.get("vote_average"),
                }
            )
    elif isinstance(data, list):
        raw_items = []
        for m in data:
            tmdb_id = m.get("tmdb_id") or m.get("id")
            title = (m.get("title") or "").strip()
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": m.get("poster_url"),
                    "release_date": m.get("release_date", ""),
                    "vote_average": m.get("vote_average"),
                }
            )
    else:
        return [], []

    matched = [x for x in raw_items if keyword_l in x["title"].lower()]
    final_list = matched if matched else raw_items

    suggestions = []
    for x in final_list[:10]:
        year = (x.get("release_date") or "")[:4]
        label = f"{x['title']} ({year})" if year else x["title"]
        suggestions.append((label, x["tmdb_id"]))

    cards = final_list[:limit]
    return suggestions, cards


# =============================================================================
# UI COMPONENTS
# =============================================================================
def movie_card(card: dict, key_prefix: str, idx: int):
    tmdb_id = card.get("tmdb_id")
    title = card.get("title", "Untitled")
    poster = card.get("poster_url") or PLACEHOLDER_POSTER
    rating = card.get("vote_average")
    year = (card.get("release_date") or "")[:4]
    match_pct = card.get("match_pct")

    rating_html = f'<div class="rating-badge">★ {rating:.1f}</div>' if rating else ""
    match_html = f'<div class="match-pill">{match_pct}% match</div>' if match_pct else ""

    st.markdown(
        f"""
        <div class="movie-card">
            <div class="poster-wrap">
                <img src="{poster}" alt="{title}" />
                {rating_html}
            </div>
            <div class="card-body">
                <div class="card-title">{title}</div>
                <div class="card-year">{year or '&nbsp;'}</div>
                {match_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View details", key=f"{key_prefix}_{idx}_{tmdb_id}", use_container_width=True):
        if tmdb_id:
            goto_details(tmdb_id)


def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies to show.")
        return
    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            with colset[c]:
                movie_card(cards[idx], key_prefix, idx)
            idx += 1


def section_header(title: str, tag: str = ""):
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-title">{title}</div>
            <div class="section-tag">{tag}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("Content-based movie discovery")
    if st.button("🏠 Home", use_container_width=True):
        goto_home()

    st.markdown("---")
    st.markdown("**Browse**")
    home_category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        index=0,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    grid_cols = st.slider("Grid columns", 4, 8, 6)

    st.markdown("---")
    st.caption("How recommendations work")
    st.caption(
        "TF-IDF over each movie's overview + genres + tagline (unigrams & "
        "bigrams), ranked by cosine similarity, re-ranked with a popularity/"
        "vote-count signal. The 'describe a vibe' box vectorizes your free "
        "text in the same TF-IDF space — no separate embedding model."
    )

# =============================================================================
# HERO
# =============================================================================
st.markdown(
    """
    <div class="hero">
        <span class="hero-eyebrow">NLP-powered recommendations</span>
        <div class="hero-title">Find your next favorite movie</div>
        <div class="hero-sub">
            Search a title to get plot-similar picks from a TF-IDF model trained
            on 45k+ movies, alongside genre-matched trending picks.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

typed = st.text_input(
    "Search",
    placeholder="Search a movie — e.g. Inception, Titanic, Interstellar...",
    label_visibility="collapsed",
)

st.markdown('<div class="vibe-label">— or describe what you\'re in the mood for —</div>', unsafe_allow_html=True)

if "vibe_query" not in st.session_state:
    st.session_state.vibe_query = ""

with st.form("vibe_form", clear_on_submit=False):
    vibe_col, btn_col = st.columns([5, 1])
    with vibe_col:
        vibe_text = st.text_area(
            "Vibe search",
            placeholder="e.g. a time-loop heist where the crew doesn't trust each other...",
            height=68,
            label_visibility="collapsed",
        )
    with btn_col:
        vibe_submitted = st.form_submit_button("🔮 Find it", use_container_width=True)

if vibe_submitted and vibe_text.strip():
    if len(vibe_text.strip()) < 3:
        st.caption("Give it a few more words to work with.")
    else:
        st.session_state.vibe_query = vibe_text.strip()
        st.session_state.view = "home"

# =============================================================================
# VIEW: HOME
# =============================================================================
if st.session_state.view == "home":
    if st.session_state.vibe_query:
        query = st.session_state.vibe_query
        if st.button("✕ Clear vibe search"):
            st.session_state.vibe_query = ""
            st.rerun()

        data, err = api_get_json("/recommend/vibe", params={"query": query, "top_n": 18})
        if err or data is None:
            st.error(f"Vibe search failed: {err}")
        else:
            cards = to_cards_from_tfidf_items(data.get("results"))
            section_header(f'Matches for "{query}"', "TF-IDF · free-text search")
            poster_grid(cards, cols=grid_cols, key_prefix="vibe_results")
        st.stop()

    if typed.strip():
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters for suggestions.")
        else:
            data, err = api_get_json("/tmdb/search", params={"query": typed.strip()})
            if err or data is None:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_tmdb_search_to_cards(data, typed.strip(), limit=24)

                if suggestions:
                    labels = ["-- Jump to a movie --"] + [s[0] for s in suggestions]
                    selected = st.selectbox("Suggestions", labels, index=0)
                    if selected != "-- Jump to a movie --":
                        label_to_id = {s[0]: s[1] for s in suggestions}
                        goto_details(label_to_id[selected])
                else:
                    st.info("No suggestions found. Try another keyword.")

                section_header(f'Results for "{typed.strip()}"', f"{len(cards)} found")
                poster_grid(cards, cols=grid_cols, key_prefix="search_results")
        st.stop()

    section_header(f"{home_category.replace('_', ' ').title()} now", "Live from TMDB")
    home_cards, err = api_get_json("/home", params={"category": home_category, "limit": 24})
    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()
    poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")

# =============================================================================
# VIEW: DETAILS
# =============================================================================
elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    if st.button("← Back to Home"):
        goto_home()

    data, err = api_get_json(f"/movie/id/{tmdb_id}")
    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    left, right = st.columns([1, 2.4], gap="large")

    with left:
        poster = data.get("poster_url") or PLACEHOLDER_POSTER
        st.markdown(
            f'<div class="movie-card"><div class="poster-wrap"><img src="{poster}" /></div></div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="detail-card">', unsafe_allow_html=True)
        st.markdown(f"## {data.get('title', '')}")
        release = data.get("release_date") or "—"
        rating = data.get("vote_average")
        rating_str = f"★ {rating:.1f}/10" if rating else "—"
        st.markdown(
            f'<div class="meta-row">📅 {release} &nbsp;·&nbsp; {rating_str}</div>',
            unsafe_allow_html=True,
        )
        genres = data.get("genres", [])
        if genres:
            chips = "".join(f'<span class="genre-chip">{g["name"]}</span>' for g in genres)
            st.markdown(f"<div style='margin: 10px 0;'>{chips}</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Overview**")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    if data.get("backdrop_url"):
        st.markdown(
            f"""<div style="margin-top:20px; border-radius:16px; overflow:hidden;">
            <img src="{data['backdrop_url']}" style="width:100%; display:block;" /></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    title = (data.get("title") or "").strip()
    if title:
        bundle, err2 = api_get_json(
            "/movie/search", params={"query": title, "tfidf_top_n": 12, "genre_limit": 12}
        )
        if not err2 and bundle:
            section_header("Similar plots & themes", "TF-IDF · NLP model")
            poster_grid(
                to_cards_from_tfidf_items(bundle.get("tfidf_recommendations")),
                cols=grid_cols,
                key_prefix="details_tfidf",
            )
            section_header("More like this", "Genre match · TMDB")
            poster_grid(
                bundle.get("genre_recommendations", []),
                cols=grid_cols,
                key_prefix="details_genre",
            )
        else:
            section_header("More like this", "Genre match (fallback)")
            genre_only, err3 = api_get_json("/recommend/genre", params={"tmdb_id": tmdb_id, "limit": 18})
            if not err3 and genre_only:
                poster_grid(genre_only, cols=grid_cols, key_prefix="details_genre_fallback")
            else:
                st.warning("No recommendations available right now.")
    else:
        st.warning("No title available to compute recommendations.")

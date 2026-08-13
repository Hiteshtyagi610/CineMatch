import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from common.preprocess import clean_text, parse_genres  # noqa: E402

DATA_PATH = BASE_DIR / "data" / "movies_metadata.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.drop_duplicates().reset_index(drop=True)

    keep = ["title", "overview", "genres", "tagline", "vote_count", "vote_average", "popularity"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    df = df.dropna(subset=["title"]).reset_index(drop=True)
    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")
    df["genres_parsed"] = df["genres"].apply(parse_genres)

    # Genres weighted 2x in the bag-of-words so a movie's genre pulls its
    # weight against a full paragraph overview when TF-IDF scores similarity.
    df["tags"] = (
        df["overview"] + " " + df["genres_parsed"] + " " + df["genres_parsed"] + " " + df["tagline"]
    )
    df["tags"] = df["tags"].apply(clean_text)

    # Drop rows that ended up with no usable text after cleaning
    df = df[df["tags"].str.strip().str.len() > 0].reset_index(drop=True)
    return df


def build_tfidf(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df["tags"])
    return vectorizer, matrix


def main():
    print(f"Loading raw data from {DATA_PATH} ...")
    df = load_and_clean(DATA_PATH)
    print(f"Cleaned dataset: {len(df):,} movies")

    print("Fitting TF-IDF vectorizer (unigrams + bigrams, 50k features max) ...")
    vectorizer, matrix = build_tfidf(df)
    print(f"TF-IDF matrix shape: {matrix.shape}")

    # Some titles repeat (remakes, re-releases). Keep the first occurrence
    # per title so `indices[title]` always returns a single scalar index.
    indices = pd.Series(df.index, index=df["title"])
    indices = indices[~indices.index.duplicated(keep="first")]

    slim_df = df[["title", "overview", "genres_parsed", "vote_count", "vote_average", "popularity"]].rename(
        columns={"genres_parsed": "genres"}
    )

    print(f"Saving artifacts to {MODELS_DIR} ...")
    slim_df.to_pickle(MODELS_DIR / "df.pkl")
    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODELS_DIR / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(matrix, f)
    with open(MODELS_DIR / "title_indices.pkl", "wb") as f:
        pickle.dump(indices, f)

    print("Done. Sanity check on a sample title:")
    sample_title = slim_df["title"].iloc[0]
    idx = indices[sample_title]
    query_vec = matrix[idx]
    scores = (matrix @ query_vec.T).toarray().ravel()
    top = np.argpartition(-scores, 6)[:6]
    top = top[np.argsort(-scores[top])]
    top = [i for i in top if i != idx][:5]
    print(f"  '{sample_title}' -> {slim_df['title'].iloc[top].tolist()}")


if __name__ == "__main__":
    main()

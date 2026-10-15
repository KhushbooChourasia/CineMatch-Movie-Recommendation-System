
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101522, #202039);
}
h1, h2, h3 {
    color: #e8c5ff;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = "/content/movielens.zip"
    folder = "/content/movielens"

    if not os.path.exists(
        folder + "/ml-latest-small/movies.csv"
    ):
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(folder)

    base = folder + "/ml-latest-small"

    movies = pd.read_csv(base + "/movies.csv")
    ratings = pd.read_csv(base + "/ratings.csv")

    movies["genres"] = movies["genres"].fillna("")
    ratings = ratings.drop_duplicates(["userId", "movieId"])

    return movies, ratings


@st.cache_resource
def build_models(movies, ratings):
    tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
    genre_matrix = tfidf.fit_transform(movies["genres"])
    content_sim = cosine_similarity(genre_matrix)

    user_movie = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    movie_user = user_movie.T.fillna(0)
    collab_sim = cosine_similarity(movie_user)

    collab_df = pd.DataFrame(
        collab_sim,
        index=movie_user.index,
        columns=movie_user.index
    )

    rating_summary = ratings.groupby("movieId").agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count")
    )

    return content_sim, collab_df, rating_summary


def get_recommendations(
    title, method, movies, content_sim, collab_df, top_n
):
    matches = movies.index[movies["title"] == title]

    if len(matches) == 0:
        return pd.DataFrame()

    idx = matches[0]
    movie_id = movies.loc[idx, "movieId"]

    if method == "Content-Based":
        scores = list(enumerate(content_sim[idx]))
        scores = [(i, s) for i, s in scores if i != idx]
        scores.sort(key=lambda x: x[1], reverse=True)

        rows = []
        for i, score in scores[:top_n]:
            rows.append({
                "movieId": movies.iloc[i]["movieId"],
                "Movie": movies.iloc[i]["title"],
                "Genres": movies.iloc[i]["genres"],
                "Similarity": round(float(score), 3)
            })

        return pd.DataFrame(rows)

    if movie_id not in collab_df.index:
        return pd.DataFrame()

    scores = collab_df[movie_id].drop(
        index=movie_id
    ).sort_values(ascending=False).head(top_n)

    lookup = movies.set_index("movieId")
    rows = []

    for mid, score in scores.items():
        if mid in lookup.index:
            movie = lookup.loc[mid]
            rows.append({
                "movieId": mid,
                "Movie": movie["title"],
                "Genres": movie["genres"],
                "Similarity": round(float(score), 3)
            })

    return pd.DataFrame(rows)


st.title("🎬 CineMatch")
st.subheader("Discover your next favourite movie")
st.write(
    "A movie recommendation system powered by machine learning."
)

try:
    with st.spinner("Loading dataset and building models..."):
        movies, ratings = load_data()
        content_sim, collab_df, rating_summary = build_models(
            movies, ratings
        )
except Exception as e:
    st.error(f"Unable to load the dataset: {e}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("🎞️ Movies", f"{movies['movieId'].nunique():,}")
c2.metric("👥 Users", f"{ratings['userId'].nunique():,}")
c3.metric("⭐ Ratings", f"{len(ratings):,}")

st.divider()

movie_options = sorted(movies["title"].unique())

selected_movie = st.selectbox(
    "Search and select a movie",
    movie_options,
    index=(
        movie_options.index("Toy Story (1995)")
        if "Toy Story (1995)" in movie_options else 0
    )
)

method = st.radio(
    "Recommendation method",
    ["Content-Based", "Collaborative Filtering"],
    horizontal=True
)

# Map the UI choice to the internal method name
internal_method = (
    "Content-Based"
    if method == "Content-Based"
    else "Collaborative"
)

top_n = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

movie_row = movies[movies["title"] == selected_movie].iloc[0]
movie_id = movie_row["movieId"]

st.markdown(f"### 🍿 {selected_movie}")
st.write("**Genres:**", movie_row["genres"].replace("|", ", "))

if movie_id in rating_summary.index:
    avg = rating_summary.loc[movie_id, "average_rating"]
    count = int(rating_summary.loc[movie_id, "rating_count"])
    st.write(f"⭐ Average rating: **{avg:.2f}/5** ({count} ratings)")

if st.button("✨ Recommend Movies", type="primary"):
    with st.spinner("Finding similar movies..."):
        results = get_recommendations(
            selected_movie,
            internal_method,
            movies,
            content_sim,
            collab_df,
            top_n
        )

    st.markdown("## Your Recommendations")

    if results.empty:
        st.warning("No recommendations found.")
    else:
        results = results.merge(
            rating_summary,
            on="movieId",
            how="left"
        )

        results["average_rating"] = results[
            "average_rating"
        ].round(2)

        results = results.rename(columns={
            "average_rating": "Average Rating",
            "rating_count": "Number of Ratings"
        })

        st.dataframe(
            results.drop(columns=["movieId"]),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download recommendations",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="movie_recommendations.csv",
            mime="text/csv"
        )

st.divider()
st.caption("Built with Python • Pandas • Scikit-learn • Streamlit")

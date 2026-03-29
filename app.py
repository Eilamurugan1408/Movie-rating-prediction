# app.py — Fixed: both prediction and recommendations show at the same time

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Rating Predictor",
    page_icon="🎬",
    layout="centered"
)

# ── Load Assets ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoder():
    with open("models/genre_encoder.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("data/processed.csv")

model   = load_model()
encoder = load_encoder()
df      = load_data()

# ── Session State Setup ───────────────────────────────────────────────────────
# This is the KEY fix — stores results so they survive page reruns
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_label" not in st.session_state:
    st.session_state.prediction_label = None
if "recommendation_df" not in st.session_state:
    st.session_state.recommendation_df = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.info(
    "This app predicts how much a user will enjoy a specific movie "
    "based on historical rating patterns."
)
st.sidebar.markdown("**Dataset:** MovieLens-style ratings")
st.sidebar.markdown("**Model:** Best of LR / RF / GB / KNN")

# ── Main Header ───────────────────────────────────────────────────────────────
st.title("🎬 Movie Rating Predictor")
st.markdown("Enter your user ID and select a movie to get a predicted rating.")

# ── Input Section ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    user_id = st.number_input(
        "Your User ID",
        min_value=int(df["user_id"].min()),
        max_value=int(df["user_id"].max()),
        value=1, step=1
    )

with col2:
    movie_list = df[["movie_id", "title"]].drop_duplicates().sort_values("title")
    selected_title = st.selectbox("Select a Movie", movie_list["title"].tolist())
    selected_movie_id = movie_list[
        movie_list["title"] == selected_title
    ]["movie_id"].values[0]

# ── Feature Builder ───────────────────────────────────────────────────────────
def build_features(user_id, movie_id):
    user_ratings = df[df["user_id"] == user_id]["rating"]
    user_avg     = user_ratings.mean() if len(user_ratings) > 0 else df["rating"].mean()
    user_count   = len(user_ratings) if len(user_ratings) > 0 else 1

    movie_ratings = df[df["movie_id"] == movie_id]["rating"]
    movie_avg     = movie_ratings.mean() if len(movie_ratings) > 0 else df["rating"].mean()

    movie_genre_raw = df[df["movie_id"] == movie_id]["genre"]
    genre_str       = movie_genre_raw.values[0] if len(movie_genre_raw) > 0 else df["genre"].mode()[0]
    genre_encoded   = encoder.transform([genre_str])[0]

    return pd.DataFrame([{
        "user_id":           user_id,
        "movie_id":          movie_id,
        "genre_encoded":     genre_encoded,
        "user_avg_rating":   user_avg,
        "movie_avg_rating":  movie_avg,
        "user_rating_count": user_count
    }])

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🔮 Predict Rating", type="primary", use_container_width=True):
    features   = build_features(user_id, selected_movie_id)
    prediction = float(np.clip(model.predict(features)[0], 1.0, 5.0))

    # Save to session state so it stays visible after recommendations load
    st.session_state.prediction_result = prediction
    st.session_state.prediction_label  = selected_title

# ── Always Show Prediction If It Exists ──────────────────────────────────────
# This block runs on EVERY page load — so result never disappears
if st.session_state.prediction_result is not None:
    st.divider()
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.metric(
            label=f"Predicted Rating for '{st.session_state.prediction_label}'",
            value=f"⭐ {st.session_state.prediction_result:.2f} / 5.00"
        )

    score = st.session_state.prediction_result
    if score >= 4.5:
        st.success("🎉 You'll probably love this movie!")
    elif score >= 3.5:
        st.info("👍 This looks like a solid watch for you.")
    elif score >= 2.5:
        st.warning("🤔 You might enjoy it, but it's not a perfect match.")
    else:
        st.error("👎 This movie may not be your style.")

# ── Recommendations Section ───────────────────────────────────────────────────
st.divider()
st.subheader("🏆 Top Movie Recommendations For You")

n_recommend = st.slider("How many recommendations?", min_value=3, max_value=10, value=5)

if st.button("🎯 Get My Recommendations", use_container_width=True):
    all_movies  = df[["movie_id", "title", "genre"]].drop_duplicates()
    predictions = []

    progress = st.progress(0, text="Finding best movies for you...")
    total    = len(all_movies)

    for i, (_, row) in enumerate(all_movies.iterrows()):
        features = build_features(user_id, row["movie_id"])
        pred     = float(np.clip(model.predict(features)[0], 1.0, 5.0))
        predictions.append({
            "Movie":            row["title"],
            "Genre":            row["genre"],
            "Predicted Rating": round(pred, 2)
        })
        # Update progress bar every 50 movies
        if i % 50 == 0:
            progress.progress(int((i / total) * 100),
                              text=f"Analysing movies... {i}/{total}")

    progress.progress(100, text="Done!")

    top_n = (pd.DataFrame(predictions)
               .sort_values("Predicted Rating", ascending=False)
               .head(n_recommend)
               .reset_index(drop=True))
    top_n.index += 1

    # Save to session state
    st.session_state.recommendation_df = top_n

# ── Always Show Recommendations If They Exist ─────────────────────────────────
if st.session_state.recommendation_df is not None:
    st.dataframe(st.session_state.recommendation_df, use_container_width=True)
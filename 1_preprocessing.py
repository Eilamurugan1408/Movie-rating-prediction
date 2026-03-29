# 1_preprocessing.py — Fixed for real MovieLens dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/ratings.csv")
print(f"Loaded dataset: {df.shape[0]} rows")
print(df.head())

# ── 2. Rename columns to snake_case (fixes the KeyError) ─────────────────────
# MovieLens uses 'userId' / 'movieId' — we standardise to 'user_id' / 'movie_id'
df.rename(columns={"userId": "user_id", "movieId": "movie_id"}, inplace=True)
print("\nColumns after rename:", df.columns.tolist())

# ── 3. Check if we have a movies.csv for titles and genres ───────────────────
# MovieLens datasets come with a separate movies.csv file
movies_path = "data/movies.csv"
if os.path.exists(movies_path):
    movies = pd.read_csv(movies_path)
    movies.rename(columns={"movieId": "movie_id"}, inplace=True)

    # Extract the FIRST genre only (movies can have multiple, e.g. "Action|Comedy")
    movies["genre"] = movies["genres"].str.split("|").str[0]
    movies["genre"] = movies["genre"].replace("(no genres listed)", "Unknown")

    # Merge ratings with movie metadata
    df = df.merge(movies[["movie_id", "title", "genre"]], on="movie_id", how="left")
    print(f"\nMerged with movies.csv — columns: {df.columns.tolist()}")
else:
    # No movies.csv — create placeholder columns so the rest of the code works
    print("\nNo movies.csv found — using placeholder genre and title.")
    df["genre"] = "Unknown"
    df["title"] = "Movie_" + df["movie_id"].astype(str)

# ── 4. Handle Missing Values ──────────────────────────────────────────────────
df.dropna(subset=["rating"], inplace=True)
df["genre"].fillna("Unknown", inplace=True)
df["title"].fillna("Movie_" + df["movie_id"].astype(str), inplace=True)

print(f"\nMissing values after cleanup:\n{df.isnull().sum()}")

# ── 5. Feature Engineering ────────────────────────────────────────────────────
# User's average rating — captures if a user is generous or harsh
user_avg = df.groupby("user_id")["rating"].mean().rename("user_avg_rating")
df = df.join(user_avg, on="user_id")

# Movie's average rating — captures overall movie quality
movie_avg = df.groupby("movie_id")["rating"].mean().rename("movie_avg_rating")
df = df.join(movie_avg, on="movie_id")

# How many ratings a user has given — captures user activity level
user_count = df.groupby("user_id")["rating"].count().rename("user_rating_count")
df = df.join(user_count, on="user_id")

# ── 6. Encode Genre ───────────────────────────────────────────────────────────
le_genre = LabelEncoder()
df["genre_encoded"] = le_genre.fit_transform(df["genre"])

with open("models/genre_encoder.pkl", "wb") as f:
    pickle.dump(le_genre, f)
print(f"\nGenres found: {le_genre.classes_.tolist()}")

# ── 7. Train/Test Split ───────────────────────────────────────────────────────
feature_cols = [
    "user_id", "movie_id", "genre_encoded",
    "user_avg_rating", "movie_avg_rating", "user_rating_count"
]

X = df[feature_cols]
y = df["rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")

# ── 8. Save Everything ────────────────────────────────────────────────────────
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv",   index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",   index=False)
df.to_csv("data/processed.csv",    index=False)

print("\nPreprocessing complete. All files saved to data/")
print(f"Final dataframe shape: {df.shape}")

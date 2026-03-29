# 2_eda.py — Split into 3 separate figure windows for clarity

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed.csv")
sns.set_theme(style="whitegrid", palette="muted")

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Rating Distribution + Ratings per User
# ══════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle("Figure 1 — Rating Overview", fontsize=16, fontweight="bold")

# Chart 1: Rating Distribution
sns.histplot(df["rating"], bins=9, kde=True,
             ax=axes1[0], color="#5C85D6", edgecolor="white")
axes1[0].set_title("Rating Distribution", fontsize=13)
axes1[0].set_xlabel("Rating")
axes1[0].set_ylabel("Count")
mean_val = df["rating"].mean()
axes1[0].axvline(mean_val, color="red", linestyle="--",
                 linewidth=1.5, label=f"Mean = {mean_val:.2f}")
axes1[0].legend()

# Chart 2: Ratings per User
user_counts = df.groupby("user_id")["rating"].count()
sns.histplot(user_counts, bins=30, ax=axes1[1],
             color="#E4845A", edgecolor="white")
axes1[1].set_title("Ratings per User", fontsize=13)
axes1[1].set_xlabel("Number of Ratings Given")
axes1[1].set_ylabel("Number of Users")
axes1[1].axvline(user_counts.median(), color="navy", linestyle="--",
                 linewidth=1.5, label=f"Median = {user_counts.median():.0f}")
axes1[1].legend()

plt.tight_layout(pad=3.0)
plt.savefig("data/fig1_rating_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Genre Analysis (the congested ones — now given full space)
# ══════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8))
fig2.suptitle("Figure 2 — Genre Analysis", fontsize=16, fontweight="bold")

# Chart 3: Average Rating by Genre — HORIZONTAL bar (no label overlap!)
genre_avg = df.groupby("genre")["rating"].mean().sort_values(ascending=True)
colors = ["#5C85D6" if v >= genre_avg.mean() else "#B0C4DE"
          for v in genre_avg.values]
genre_avg.plot(kind="barh", ax=axes2[0], color=colors, edgecolor="white")
axes2[0].set_title("Average Rating by Genre", fontsize=13)
axes2[0].set_xlabel("Average Rating")
axes2[0].set_ylabel("Genre")
axes2[0].axvline(genre_avg.mean(), color="red", linestyle="--",
                 linewidth=1.5, label=f"Overall avg: {genre_avg.mean():.2f}")
axes2[0].legend()
# Add value labels on bars
for i, val in enumerate(genre_avg.values):
    axes2[0].text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=9)

# Chart 4: Box plot — HORIZONTAL so genre names are readable
top_genres = df["genre"].value_counts().head(10).index
df_top = df[df["genre"].isin(top_genres)]
sns.boxplot(data=df_top, y="genre", x="rating",
            palette="pastel", ax=axes2[1], orient="h")
axes2[1].set_title("Rating Spread by Genre (Top 10)", fontsize=13)
axes2[1].set_xlabel("Rating")
axes2[1].set_ylabel("Genre")
axes2[1].axvline(df["rating"].mean(), color="red", linestyle="--",
                 linewidth=1.5, label=f"Overall avg: {df['rating'].mean():.2f}")
axes2[1].legend()

plt.tight_layout(pad=3.0)
plt.savefig("data/fig2_genre_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Correlation Heatmap + User vs Movie Avg Scatter
# ══════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
fig3.suptitle("Figure 3 — Feature Relationships", fontsize=16, fontweight="bold")

# Chart 5: Correlation Heatmap
import numpy as np
numeric_cols = ["rating", "user_avg_rating", "movie_avg_rating", "user_rating_count"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            ax=axes3[0], linewidths=1, square=True,
            vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8})
axes3[0].set_title("Feature Correlation Heatmap", fontsize=13)

# Chart 6: Scatter plot
sample = df.sample(5000, random_state=42)
scatter = axes3[1].scatter(sample["user_avg_rating"],
                           sample["movie_avg_rating"],
                           c=sample["rating"],
                           cmap="RdYlGn",
                           alpha=0.5, s=15,
                           vmin=1, vmax=5)
plt.colorbar(scatter, ax=axes3[1], label="Actual Rating")
axes3[1].set_title("User Avg vs Movie Avg Rating", fontsize=13)
axes3[1].set_xlabel("User Avg Rating (user bias)")
axes3[1].set_ylabel("Movie Avg Rating (movie quality)")
axes3[1].axhline(df["movie_avg_rating"].mean(), color="gray",
                 linestyle="--", linewidth=0.8, alpha=0.6)
axes3[1].axvline(df["user_avg_rating"].mean(), color="gray",
                 linestyle="--", linewidth=0.8, alpha=0.6)

plt.tight_layout(pad=3.0)
plt.savefig("data/fig3_feature_relationships.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nAll 3 figures shown and saved to data/ folder.")
print("fig1_rating_overview.png")
print("fig2_genre_analysis.png")
print("fig3_feature_relationships.png")
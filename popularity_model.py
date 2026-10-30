"""
Weighted Bayesian Popularity Baseline.
"""
import numpy as np
import pandas as pd

class PopularityRecommender:
    """
    Bayesian-damped IMDB-style ranking model:
    Weighted Rating = (v / (v + m)) * R + (m / (v + m)) * C
    v: number of ratings for movie
    m: minimum rating count threshold (quantile 70)
    R: average rating of movie
    C: global mean rating
    """
    def __init__(self, m_quantile=0.70):
        self.m_quantile = m_quantile
        self.ranked_items = None
        self.global_mean = 0.0

    def fit(self, train_df, items_df):
        self.global_mean = train_df["rating"].mean()
        stats = train_df.groupby("movie_id")["rating"].agg(["count", "mean"]).reset_index()
        m = stats["count"].quantile(self.m_quantile)
        
        def bayesian_score(row):
            v = row["count"]
            R = row["mean"]
            return (v / (v + m)) * R + (m / (v + m)) * self.global_mean

        stats["score"] = stats.apply(bayesian_score, axis=1)
        ranked = stats.merge(items_df[["movie_id", "title", "genres_str"]], on="movie_id")
        self.ranked_items = ranked.sort_values(by="score", ascending=False).reset_index(drop=True)
        return self

    def recommend(self, user_train_ratings, top_k=10):
        seen_movies = set(user_train_ratings["movie_id"].tolist())
        candidates = self.ranked_items[~self.ranked_items["movie_id"].isin(seen_movies)]
        return candidates.head(top_k)[["movie_id", "title", "genres_str", "score"]]
"""
Content-Based Filtering via Genre TF-IDF Vectorization and User Profile Centroid Modeling.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
        self.tfidf_matrix = None
        self.items_df = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}

    def fit(self, items_df):
        self.items_df = items_df.copy().reset_index(drop=True)
        self.movie_id_to_idx = {row["movie_id"]: idx for idx, row in self.items_df.iterrows()}
        self.idx_to_movie_id = {idx: row["movie_id"] for idx, row in self.items_df.iterrows()}
        
        # Fit TF-IDF on genre text
        self.tfidf_matrix = self.vectorizer.fit_transform(self.items_df["genres_str"]).toarray()
        return self

    def build_user_profile(self, user_ratings_df):
        """
        Computes a rating-weighted genre profile vector:
        Profile_u = sum( (rating_i - 2.5) * ItemVector_i )
        """
        valid_ratings = user_ratings_df[user_ratings_df["movie_id"].isin(self.movie_id_to_idx)]
        if valid_ratings.empty:
            return np.zeros(self.tfidf_matrix.shape[1])
        
        profile = np.zeros(self.tfidf_matrix.shape[1])
        weight_sum = 0.0
        
        for _, row in valid_ratings.iterrows():
            idx = self.movie_id_to_idx[row["movie_id"]]
            # Centered rating weight: 5 -> +2.5, 4 -> +1.5, 1 -> -1.5
            weight = row["rating"] - 2.5
            profile += weight * self.tfidf_matrix[idx]
            weight_sum += abs(weight)
            
        if weight_sum > 0:
            profile /= weight_sum
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile /= norm
                
        return profile

    def predict_all_for_user(self, user_ratings_df, candidate_movie_ids):
        user_vector = self.build_user_profile(user_ratings_df).reshape(1, -1)
        if np.linalg.norm(user_vector) == 0:
            return {mid: 0.5 for mid in candidate_movie_ids}
        
        valid_candidates = [mid for mid in candidate_movie_ids if mid in self.movie_id_to_idx]
        candidate_indices = [self.movie_id_to_idx[mid] for mid in valid_candidates]
        candidate_vectors = self.tfidf_matrix[candidate_indices]
        
        sims = cosine_similarity(user_vector, candidate_vectors).flatten()
        # Scale cosine similarities [-1, 1] linearly into [0, 1]
        scaled_sims = (sims + 1.0) / 2.0
        
        res = {mid: float(scaled_sims[k]) for k, mid in enumerate(valid_candidates)}
        for mid in candidate_movie_ids:
            if mid not in res:
                res[mid] = 0.5
        return res
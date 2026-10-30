"""
Dynamic Weighted Hybrid Recommender with min-max score calibration and cold-start fallback.
"""
import numpy as np
import pandas as pd
from config import DEFAULT_ALPHA, COLD_START_THRESHOLD

class HybridRecommender:
    def __init__(self, cf_model, cb_model, alpha=DEFAULT_ALPHA, cold_start_th=COLD_START_THRESHOLD):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.alpha = alpha
        self.cold_start_th = cold_start_th

    def recommend(self, user_id, user_train_ratings, candidate_items_df, top_k=10):
        candidate_ids = candidate_items_df["movie_id"].tolist()
        num_interactions = len(user_train_ratings)
        
        # Adaptive weight fallback: if user is cold-start, rely primarily on content signals
        effective_alpha = self.alpha if num_interactions >= self.cold_start_th else 0.05
        
        # Step 1: Collaborative Filtering Scores (Range: [1.0, 5.0])
        cf_raw_scores = self.cf_model.predict_all_for_user(user_id, candidate_ids)
        # Normalize CF ratings to [0.0, 1.0]: S_CF = (r - 1.0) / 4.0
        cf_norm = {mid: (score - 1.0) / 4.0 for mid, score in cf_raw_scores.items()}
        
        # Step 2: Content-Based Scores (Range: [0.0, 1.0])
        cb_norm = self.cb_model.predict_all_for_user(user_train_ratings, candidate_ids)
        
        # Step 3: Calibrated Convex Combination
        # Hybrid Score = alpha * CF_norm + (1 - alpha) * CB_norm
        results = []
        for mid in candidate_ids:
            s_cf = cf_norm.get(mid, 0.5)
            s_cb = cb_norm.get(mid, 0.5)
            hybrid_score = (effective_alpha * s_cf) + ((1.0 - effective_alpha) * s_cb)
            results.append({
                "movie_id": mid,
                "score": hybrid_score,
                "cf_score": s_cf,
                "cb_score": s_cb,
                "pred_rating": 1.0 + (hybrid_score * 4.0)
            })
            
        ranked_df = pd.DataFrame(results).sort_values(by="score", ascending=False)
        merged = ranked_df.merge(candidate_items_df, on="movie_id")
        
        return merged.head(top_k), effective_alpha
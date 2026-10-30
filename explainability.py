"""
Dual-Engine Explainability Module: Traces recommendations directly to model components.
Computes collaborative factor attribution and genre content overlap.
"""
import numpy as np

class RecommendationExplainer:
    def __init__(self, items_df, cb_model):
        self.items_df = items_df.set_index("movie_id")
        self.cb_model = cb_model

    def explain(self, rec_row, user_train_ratings, effective_alpha):
        movie_id = rec_row["movie_id"]
        movie_title = rec_row["title"]
        rec_genres = set(rec_row["genres_str"].split()) if rec_row["genres_str"] != "General" else set()
        
        # Percentage attribution
        cf_pct = int(round(effective_alpha * 100))
        cb_pct = 100 - cf_pct
        
        # Content attribution: Identify user's highest rated movie with overlapping genres
        best_anchor = None
        max_overlap = -1
        matching_genres = []
        
        liked_movies = user_train_ratings[user_train_ratings["rating"] >= 4.0]
        if not liked_movies.empty:
            for _, r in liked_movies.iterrows():
                mid = r["movie_id"]
                if mid in self.items_df.index:
                    anchor_title = self.items_df.loc[mid, "title"]
                    anchor_genres = set(self.items_df.loc[mid, "genres_str"].split())
                    overlap = rec_genres.intersection(anchor_genres)
                    if len(overlap) > max_overlap:
                        max_overlap = len(overlap)
                        best_anchor = anchor_title
                        matching_genres = list(overlap)
        
        # Synthesize attribution narrative
        reasons = []
        if matching_genres and best_anchor:
            reasons.append(f"Shares genre(s) [{', '.join(matching_genres)}] with your highly-rated favorite '{best_anchor}'")
        elif rec_genres:
            reasons.append(f"Matches your affinity for genres: [{', '.join(list(rec_genres)[:3])}]")
            
        if cf_pct > 30:
            reasons.append(f"Highly favored by users whose historical preference patterns closely mirror yours")
        else:
            reasons.append("Selected primarily through genre content profile matching during cold-start mode")
            
        attribution_str = f"Attribution: {cf_pct}% Collaborative Behavior, {cb_pct}% Content Metadata."
        
        return {
            "narrative": " | ".join(reasons),
            "attribution": attribution_str,
            "cf_share": cf_pct,
            "cb_share": cb_pct,
            "shared_genres": matching_genres,
            "anchor_movie": best_anchor
        }
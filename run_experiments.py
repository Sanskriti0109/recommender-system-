"""
Master Experimentation Script: Runs all 4 models and outputs comparative metrics.
"""
import os
import json
import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import train_test_split_stratified
from src.popularity_model import PopularityRecommender
from src.collaborative_model import RegularizedSVD
from src.content_model import ContentBasedRecommender
from src.hybrid_model import HybridRecommender
from src.evaluation import compute_rating_error, evaluate_top_k
from config import RESULTS_DIR, DEFAULT_ALPHA

def run_all_experiments():
    print("=" * 60)
    print("STARTING RECOMMENDER EXPERIMENTAL BENCHMARK")
    print("=" * 60)
    
    ratings_df, items_df = load_raw_data()
    train_df, test_df = train_test_split_stratified(ratings_df)
    
    all_user_ids = ratings_df["user_id"].unique()
    all_movie_ids = items_df["movie_id"].unique()
    
    print(f"Dataset: MovieLens 100K | Train Ratings: {len(train_df)} | Test Ratings: {len(test_df)}")
    
    # 1. Train Baselines
    print("\n[1/4] Training Bayesian Popularity Model...")
    pop_model = PopularityRecommender().fit(train_df, items_df)
    
    print("\n[2/4] Training Latent Factor Regularized SVD...")
    svd_model = RegularizedSVD().fit(train_df, all_user_ids, all_movie_ids)
    svd_rmse, svd_mae = compute_rating_error(svd_model, test_df)
    print(f"      SVD Offline Rating Accuracy -> RMSE: {svd_rmse:.4f} | MAE: {svd_mae:.4f}")
    
    print("\n[3/4] Fitting Content-Based TF-IDF Genre Model...")
    cb_model = ContentBasedRecommender().fit(items_df)
    
    print(f"\n[4/4] Assembling Explainable Hybrid Model (alpha={DEFAULT_ALPHA})...")
    hybrid_model = HybridRecommender(svd_model, cb_model, alpha=DEFAULT_ALPHA)
    
    # Wrapper recommendation functions for evaluation
    def rec_pop(u_id, u_train, cand, top_k):
        return pop_model.recommend(u_train, top_k=top_k)
        
    def rec_svd(u_id, u_train, cand, top_k):
        preds = svd_model.predict_all_for_user(u_id, cand["movie_id"].tolist())
        ranked = cand.assign(score=cand["movie_id"].map(preds)).sort_values(by="score", ascending=False)
        return ranked.head(top_k)
        
    def rec_cb(u_id, u_train, cand, top_k):
        preds = cb_model.predict_all_for_user(u_train, cand["movie_id"].tolist())
        ranked = cand.assign(score=cand["movie_id"].map(preds)).sort_values(by="score", ascending=False)
        return ranked.head(top_k)
        
    def rec_hybrid(u_id, u_train, cand, top_k):
        recs, _ = hybrid_model.recommend(u_id, u_train, cand, top_k=top_k)
        return recs

    print("\nEvaluating Top-10 Ranking Performance across Test Users...")
    eval_results = {}
    
    print(" -> Evaluating Popularity Model...")
    eval_results["Popularity"] = evaluate_top_k(rec_pop, train_df, test_df, items_df, k=10)
    
    print(" -> Evaluating Collaborative SVD...")
    eval_results["SVD (Collaborative)"] = evaluate_top_k(rec_svd, train_df, test_df, items_df, k=10)
    
    print(" -> Evaluating Content-Based TF-IDF...")
    eval_results["Content-Based (TF-IDF)"] = evaluate_top_k(rec_cb, train_df, test_df, items_df, k=10)
    
    print(" -> Evaluating Proposed Hybrid Recommender...")
    eval_results["Hybrid (Proposed)"] = evaluate_top_k(rec_hybrid, train_df, test_df, items_df, k=10)
    
    # Ablation Study across Alpha values
    print("\nRunning Ablation Analysis over alpha in [0.0, 1.0]...")
    ablation_results = {}
    for alpha_val in [0.0, 0.2, 0.4, 0.55, 0.7, 0.85, 1.0]:
        h_ablation = HybridRecommender(svd_model, cb_model, alpha=alpha_val)
        def rec_abl(u_id, u_train, cand, top_k):
            recs, _ = h_ablation.recommend(u_id, u_train, cand, top_k=top_k)
            return recs
        res = evaluate_top_k(rec_abl, train_df, test_df, items_df, k=10, max_eval_users=80)
        ablation_results[str(alpha_val)] = res
        print(f"    Alpha={alpha_val:.2f} -> NDCG@10: {res['NDCG@10']:.4f} | Coverage: {res['Catalog_Coverage']:.4f}")

    # Save summary
    summary_df = pd.DataFrame(eval_results).T
    summary_df.to_csv(RESULTS_DIR / "model_comparison_metrics.csv")
    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=4)
        
    print("\n" + "=" * 60)
    print("FINAL EXPERIMENTAL BENCHMARK SUMMARY TABLE:")
    print("=" * 60)
    print(summary_df.to_string())
    print("\nResults successfully saved to 'results/' directory.")

if __name__ == "__main__":
    run_all_experiments()
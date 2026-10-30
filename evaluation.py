"""
Rigorous offline evaluation suite: RMSE, MAE, Precision@K, Recall@K, NDCG@K, HitRate@K,
Catalog Coverage, and Intra-List Diversity (ILD).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

def compute_rating_error(cf_model, test_df):
    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        y_true.append(row["rating"])
        y_pred.append(cf_model.predict(row["user_id"], row["movie_id"]))
        
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae

def evaluate_top_k(recommend_fn, train_df, test_df, all_items_df, k=10, relevance_th=4.0, max_eval_users=200):
    test_grouped = test_df.groupby("user_id")
    test_users = [u for u in test_grouped.groups.keys() if len(test_grouped.get_group(u)) >= 3]
    if len(test_users) > max_eval_users:
        np.random.seed(42)
        test_users = np.random.choice(test_users, size=max_eval_users, replace=False)
        
    precisions, recalls, ndcgs, hits = [], [], [], []
    recommended_items_corpus = set()
    intra_list_diversities = []
    
    # Pre-build genre feature vectors for ILD
    tfidf_matrix = all_items_df.set_index("movie_id")
    
    for u in test_users:
        u_train = train_df[train_df["user_id"] == u]
        u_test = test_df[test_df["user_id"] == u]
        relevant_test_items = set(u_test[u_test["rating"] >= relevance_th]["movie_id"].tolist())
        
        if not relevant_test_items:
            continue
            
        unrated_candidates = all_items_df[~all_items_df["movie_id"].isin(u_train["movie_id"])].copy()
        recs = recommend_fn(u, u_train, unrated_candidates, top_k=k)
        rec_ids = recs["movie_id"].tolist()
        
        recommended_items_corpus.update(rec_ids)
        
        # Precision & Recall & Hit Rate
        hits_count = sum(1 for mid in rec_ids if mid in relevant_test_items)
        precisions.append(hits_count / float(k))
        recalls.append(hits_count / float(len(relevant_test_items)))
        hits.append(1.0 if hits_count > 0 else 0.0)
        
        # NDCG@K
        dcg = 0.0
        for rank_idx, mid in enumerate(rec_ids):
            if mid in relevant_test_items:
                dcg += 1.0 / np.log2(rank_idx + 2)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant_test_items), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        
        # Intra-List Diversity (Average pairwise cosine distance between recommended items' genre vectors)
        if len(rec_ids) > 1:
            rec_genres = all_items_df[all_items_df["movie_id"].isin(rec_ids)].iloc[:, 2:21].to_numpy()
            if rec_genres.shape[0] > 1:
                cos_sim_matrix = cosine_similarity(rec_genres)
                upper_tri_indices = np.triu_indices(cos_sim_matrix.shape[0], k=1)
                pairwise_dists = 1.0 - cos_sim_matrix[upper_tri_indices]
                intra_list_diversities.append(float(np.mean(pairwise_dists)))

    total_catalog_size = len(all_items_df)
    catalog_coverage = len(recommended_items_corpus) / float(total_catalog_size)
    
    return {
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"HitRate@{k}": float(np.mean(hits)),
        "Catalog_Coverage": float(catalog_coverage),
        "Intra_List_Diversity": float(np.mean(intra_list_diversities)) if intra_list_diversities else 0.0
    }
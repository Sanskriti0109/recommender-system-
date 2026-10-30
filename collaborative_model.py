"""
Latent Factor Collaborative Filtering via Regularized Matrix Factorization (SVD).
Implemented with pure NumPy/SciPy for portability across environments.
"""
import numpy as np
import pandas as pd
from config import SVD_N_FACTORS, SVD_N_EPOCHS, SVD_LR, SVD_REG, RANDOM_SEED

class RegularizedSVD:
    def __init__(self, n_factors=SVD_N_FACTORS, n_epochs=SVD_N_EPOCHS, lr=SVD_LR, reg=SVD_REG, seed=RANDOM_SEED):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed
        
        self.global_mean = 0.0
        self.user_biases = None
        self.item_biases = None
        self.P = None  # User latent factors
        self.Q = None  # Item latent factors
        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}

    def fit(self, train_df, all_user_ids, all_movie_ids):
        np.random.seed(self.seed)
        self.user_to_idx = {uid: idx for idx, uid in enumerate(sorted(all_user_ids))}
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(sorted(all_movie_ids))}
        self.idx_to_movie = {idx: mid for mid, idx in self.movie_to_idx.items()}
        
        n_users = len(self.user_to_idx)
        n_items = len(self.movie_to_idx)
        
        self.global_mean = float(train_df["rating"].mean())
        self.user_biases = np.zeros(n_users, dtype=np.float32)
        self.item_biases = np.zeros(n_items, dtype=np.float32)
        
        # Latent factor initialization with small normal deviations
        self.P = np.random.normal(0, 0.05, (n_users, self.n_factors)).astype(np.float32)
        self.Q = np.random.normal(0, 0.05, (n_items, self.n_factors)).astype(np.float32)
        
        u_indices = np.array([self.user_to_idx[uid] for uid in train_df["user_id"]], dtype=np.int32)
        i_indices = np.array([self.movie_to_idx[mid] for mid in train_df["movie_id"]], dtype=np.int32)
        ratings = train_df["rating"].to_numpy(dtype=np.float32)
        
        # Stochastic Gradient Descent Optimization
        n_samples = len(ratings)
        indices = np.arange(n_samples)
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for idx in indices:
                u = u_indices[idx]
                i = i_indices[idx]
                r = ratings[idx]
                
                # Prediction: r_hat = mu + b_u + b_i + P_u . Q_i
                pred = self.global_mean + self.user_biases[u] + self.item_biases[i] + np.dot(self.P[u], self.Q[i])
                err = r - pred
                
                # Gradient update
                self.user_biases[u] += self.lr * (err - self.reg * self.user_biases[u])
                self.item_biases[i] += self.lr * (err - self.reg * self.item_biases[i])
                
                P_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * P_u_old - self.reg * self.Q[i])

        return self

    def predict(self, user_id, movie_id):
        # Graceful fallback for unseen entities
        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.movie_to_idx.get(movie_id)
        
        if u_idx is None and i_idx is None:
            return self.global_mean
        if u_idx is None:
            return np.clip(self.global_mean + self.item_biases[i_idx], 1.0, 5.0)
        if i_idx is None:
            return np.clip(self.global_mean + self.user_biases[u_idx], 1.0, 5.0)
            
        pred = self.global_mean + self.user_biases[u_idx] + self.item_biases[i_idx] + np.dot(self.P[u_idx], self.Q[i_idx])
        return float(np.clip(pred, 1.0, 5.0))

    def predict_all_for_user(self, user_id, candidate_movie_ids):
        u_idx = self.user_to_idx.get(user_id)
        if u_idx is None:
            return {mid: self.global_mean for mid in candidate_movie_ids}
        
        # Vectorized batch prediction for user
        valid_mids = [mid for mid in candidate_movie_ids if mid in self.movie_to_idx]
        i_indices = [self.movie_to_idx[mid] for mid in valid_mids]
        
        preds = (
            self.global_mean 
            + self.user_biases[u_idx] 
            + self.item_biases[i_indices] 
            + np.dot(self.Q[i_indices], self.P[u_idx])
        )
        clipped = np.clip(preds, 1.0, 5.0)
        
        scores = {mid: float(clipped[k]) for k, mid in enumerate(valid_mids)}
        for mid in candidate_movie_ids:
            if mid not in scores:
                scores[mid] = self.global_mean
        return scores
"""
Data preprocessing, stratified splitting, and feature preparation.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED, TEST_SIZE

def train_test_split_stratified(ratings_df, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """
    Splits ratings while ensuring every user with >= 2 ratings has representation in the training set.
    """
    train_list = []
    test_list = []
    
    grouped = ratings_df.groupby("user_id")
    for user_id, group in grouped:
        if len(group) >= 5:
            u_train, u_test = train_test_split(
                group, test_size=test_size, random_state=random_state, shuffle=True
            )
            train_list.append(u_train)
            test_list.append(u_test)
        else:
            # Users with very few interactions are preserved in train to retain interaction topology
            train_list.append(group)
            
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    return train_df, test_df

def compute_interaction_matrices(train_df, all_user_ids, all_movie_ids):
    user_to_idx = {uid: idx for idx, uid in enumerate(all_user_ids)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(all_movie_ids)}
    
    return user_to_idx, movie_to_idx
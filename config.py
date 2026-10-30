"""
Global Configuration and Hyperparameters.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset URLs
MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# Model Hyperparameters
RANDOM_SEED = 42
TEST_SIZE = 0.20
TOP_K_DEFAULT = 10
RELEVANCE_THRESHOLD = 4.0  # Ratings >= 4.0 are relevant for ranking metrics

# SVD Parameters
SVD_N_FACTORS = 32
SVD_N_EPOCHS = 25
SVD_LR = 0.007
SVD_REG = 0.02

# Hybrid Parameter
DEFAULT_ALPHA = 0.55  # Weight assigned to CF score; (1 - alpha) assigned to CB score
COLD_START_THRESHOLD = 5  # Users with fewer ratings use CB fallback
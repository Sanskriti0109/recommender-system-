"""
Automated downloading, extraction, and validation of the MovieLens 100K dataset.
Configured with SSL bypass and manual fallback support for Windows.
"""
import io
import os
import zipfile
import requests
import urllib3
import pandas as pd
from config import DATA_DIR, MOVIELENS_100K_URL

# Suppress insecure request warnings caused by Windows root CA missing bundle
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def download_and_extract_ml_100k():
    zip_path = DATA_DIR / "ml-100k.zip"
    extracted_folder = DATA_DIR / "ml-100k"
    
    # 1. If already extracted, skip
    if (extracted_folder / "u.data").exists() and (extracted_folder / "u.item").exists():
        print("MovieLens 100K dataset already present and verified.")
        return

    # 2. If zip is already placed in data/, just extract
    if zip_path.exists():
        print(f"Extracting existing archive at {zip_path}...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
        print("Extraction complete.")
        return

    # 3. Download with SSL verification bypass and browser headers
    print(f"Downloading MovieLens 100K from {MOVIELENS_100K_URL}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(MOVIELENS_100K_URL, timeout=60, verify=False, headers=headers)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(DATA_DIR)
        print("Download and extraction complete.")
    except Exception as e:
        print(f"\n[ERROR] Automated download encountered an error: {e}")
        print("\n--> MANUAL FALLBACK INSTRUCTIONS <--")
        print(f"1. Download ml-100k.zip in your browser from:")
        print(f"   https://files.grouplens.org/datasets/movielens/ml-100k.zip")
        print(f"2. Move or copy the downloaded 'ml-100k.zip' into:")
        print(f"   {DATA_DIR.resolve()}")
        print(f"3. Re-run your command.\n")
        raise e

def load_raw_data():
    download_and_extract_ml_100k()
    data_dir = DATA_DIR / "ml-100k"
    
    # Load ratings
    ratings_cols = ["user_id", "movie_id", "rating", "timestamp"]
    ratings_df = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=ratings_cols,
        engine="python"
    )
    
    # Load items (movies)
    items_cols = [
        "movie_id", "title", "release_date", "video_release_date", "IMDb_URL"
    ] + GENRE_COLS
    items_df = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        names=items_cols,
        usecols=["movie_id", "title"] + GENRE_COLS,
        encoding="latin-1",
        engine="python"
    )
    
    # Create genre strings for content filtering
    def extract_genres(row):
        active = [g for g in GENRE_COLS if row[g] == 1 and g != "unknown"]
        return " ".join(active) if active else "General"
    
    items_df["genres_str"] = items_df.apply(extract_genres, axis=1)
    
    return ratings_df, items_df

if __name__ == "__main__":
    r, i = load_raw_data()
    print(f"Loaded {len(r)} ratings across {r['user_id'].nunique()} users and {r['movie_id'].nunique()} movies.")
# src/models/predict.py

""" Prediction module for the movie recommendation system.

Responsibilities:
- Load trained model
- Generate recommendations for users
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, load_npz

from src.utils import get_logger

logger = get_logger(__name__)

MODELS_PATH = Path(__file__).parent.parent.parent / "models"
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed"

def load_model(model_name: str = "itemknn_k50") -> joblib.load:
    """
    Load trained model from models/ folder.
    
    Args:
        model_name: name of the model file
    
    Returns:
        Trained model object
    """
    model_path = MODELS_PATH / f"{model_name}.joblib"
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    else:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    return model

# load artifact data
def load_artifacts() -> tuple:
    """
    Load user-item matrix and mappings from data/ folder.
    
    Returns:
        user_item_matrix: sparse matrix
        user_map: user_id to index mapping
        item_map: movie_id to index mapping
    """

    # load matrix
    user_item_matrix = load_npz(DATA_PATH / "user_item_matrix.npz")

    # load mappings
    user_map_df = pd.read_csv(DATA_PATH / "mappings" / "user_map.csv")
    item_map_df = pd.read_csv(DATA_PATH / "mappings" / "item_map.csv")
    
    # convert to dict
    user_map = dict(zip(user_map_df["user_id"], user_map_df["user_idx"]))
    item_map = dict(zip(item_map_df["movie_id"], item_map_df["item_idx"]))
    
    # reverse item map (idx → movie_id)
    idx_to_item = dict(zip(item_map_df["item_idx"], item_map_df["movie_id"]))
    
    logger.info("Artifacts loaded successfully")
    
    return user_item_matrix, user_map, item_map, idx_to_item

# Evaluation module for the movie recommendation system.
def recommend_movies(
    user_id: int,
    model,
    user_item_matrix: csr_matrix,
    user_map: dict,
    item_map: dict,
    idx_to_item: dict,
    movies: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    """
    Generate top N movie recommendations for a user.
    
    Args:
        user_id: user to recommend for
        model: trained model
        user_item_matrix: sparse matrix
        user_map: user_id to index mapping
        item_map: movie_id to index mapping
        idx_to_item: index to movie_id mapping
        movies: movies dataframe for titles
        n: number of recommendations
    Returns:
        dataframe with top N recommendations
    """
    user_idx = user_map.get(user_id)
    if user_idx is None:
        logger.error(f"User {user_id} not found!")
        return None

    # get user ratings
    user_ratings = user_item_matrix[user_idx].toarray().flatten()

    # find unwatched movies
    unwatched = np.where(user_ratings == 0)[0]

    # predict scores for unwatched movies
    scores = []
    for item_idx in unwatched:
        distances, indices = model.kneighbors(
            user_item_matrix.T[item_idx],
            n_neighbors=model.n_neighbors
        )
        similar_ratings = user_ratings[indices[0]]
        weights = 1 - distances[0]

        if weights.sum() > 0 and similar_ratings.sum() > 0:
            pred = np.average(similar_ratings, weights=weights)
        else:
            pred = 0

        movie_id = idx_to_item.get(item_idx)
        scores.append((movie_id, pred))

    # sort by predicted score
    scores.sort(key=lambda x: x[1], reverse=True)
    top_n = scores[:n]

    # get movie titles
    recommendations = []
    for movie_id, score in top_n:
        title = movies[movies["movie_id"] == movie_id]["title"].values
        if len(title) > 0:
            recommendations.append({
                "movie_id": movie_id,
                "title": title[0],
                "predicted_score": round(score, 2)
            })

    result = pd.DataFrame(recommendations)
    logger.info(f"Generated {len(result)} recommendations for User {user_id}")
    return result

def predict_pipeline(user_id: int, movies: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Run full prediction pipeline for a user.
    
    Args:
        user_id: user to recommend for
        movies: movies dataframe for titles
        n: number of recommendations
    Returns:
        dataframe with top N recommendations
    """
    # load model and artifacts
    model = load_model()
    user_item_matrix, user_map, item_map, idx_to_item = load_artifacts()

    # generate recommendations
    recommendations = recommend_movies(
        user_id=user_id,
        model=model,
        user_item_matrix=user_item_matrix,
        user_map=user_map,
        item_map=item_map,
        idx_to_item=idx_to_item,
        movies=movies,
        n=n
    )

    return recommendations
# src/models/train.py

""" Training module for the movie recommendation system. 
This module contains functions to train the recommendation model using the training data."""

import joblib
import numpy as np
from pathlib import Path
import mlflow

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from src.utils import get_logger

logger = get_logger(__name__)

# Define the path to save the trained model
MODELS_PATH = Path(__file__).parent.parent.parent / "models"

def train_model(user_item_matrix: csr_matrix, k: int = 50) -> NearestNeighbors:
    """
    Train ItemKNN model on user-item matrix.
    
    Args:
        user_item_matrix: sparse user-item matrix
        k: number of neighbors
    Returns:
        trained NearestNeighbors model
    """
    logger.info(f"Training ItemKNN with K={k}")
    
    model = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute"
    )
    
    model.fit(user_item_matrix.T)
    
    logger.info(f"Model trained successfully!")
    
    return model

def save_model(model: NearestNeighbors, model_name: str = "itemknn") -> None:
    """
    Save trained model to models/ folder.
    
    Args:
        model: trained NearestNeighbors model
        model_name: name of the model file
    """
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    model_path = MODELS_PATH / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    
    logger.info(f"Model saved to {model_path} successfully!")


def train_pipeline(user_item_matrix: csr_matrix, k_values: list = [10, 20, 50]) -> NearestNeighbors:
    """
    Train multiple ItemKNN models with different K values,
    track with MLflow and save the best model.
    
    Args:
        user_item_matrix: sparse user-item matrix
        k_values: list of K values to try
    Returns:
        best trained model
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("movie-recommendation-system")
    
    best_model = None
    best_k = None
    
    for k in k_values:
        with mlflow.start_run(run_name=f"ItemKNN_K{k}"):
            
            # train model
            model = train_model(user_item_matrix, k=k)
            
            # log parameters
            mlflow.log_param("k", k)
            mlflow.log_param("metric", "cosine")
            mlflow.log_param("algorithm", "brute")
            
            logger.info(f"K={k} logged to MLflow successfully!")
            
            # track best model
            if best_model is None:
                best_model = model
                best_k = k
    
    # save best model
    save_model(best_model, model_name=f"itemknn_k{best_k}")
    
    logger.info(f"Best model: ItemKNN K={best_k} saved successfully!")
    
    return best_model
"""
Evaluation module for movie recommendation system.

Responsibilities:
- Evaluate best model on full test set
- Calculate RMSE, Precision@K, Recall@K
- Save evaluation report
- Log metrics to MLflow
"""

import json
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import root_mean_squared_error,mean_squared_error

from src.utils import get_logger

logger = get_logger(__name__)

REPORTS_PATH = Path(__file__).parent.parent.parent / "reports"

def save_evaluation_report(report: dict, report_name: str = "evaluation_report.json") -> None:
    """
    Save evaluation report to reports/ folder.
    
    Args:
        report: dictionary containing evaluation metrics
        report_name: name of the report file
    """
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    
    report_path = REPORTS_PATH / report_name
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"Evaluation report saved to {report_path} successfully!")

def calculate_rmse(
    model: NearestNeighbors,
    user_item_matrix: csr_matrix,
    test: pd.DataFrame,
    user_map: dict,
    item_map: dict
) -> float:
    """
    Calculate RMSE on full test set.
    
    Args:
        model: trained model
        user_item_matrix: sparse matrix
        test: test dataframe
        user_map: user_id to index mapping
        item_map: movie_id to index mapping
    Returns:
        RMSE score
    """
    actuals = []
    predictions = []

    for _, row in test.iterrows():
        user_idx = user_map.get(row["user_id"])
        item_idx = item_map.get(row["movie_id"])

        if user_idx is None or item_idx is None:
            continue

        distances, indices = model.kneighbors(
            user_item_matrix.T[item_idx],
            n_neighbors=model.n_neighbors
        )

        similar_ratings = user_item_matrix[user_idx, indices[0]].toarray().flatten()
        weights = 1 - distances[0]

        if weights.sum() > 0 and similar_ratings.sum() > 0:
            pred = np.average(similar_ratings, weights=weights)
        else:
            pred = 0

        actuals.append(row["rating"])
        predictions.append(pred)

    rmse = root_mean_squared_error(actuals, predictions)
    logger.info(f"RMSE on full test set: {rmse:.4f}")
    return rmse

def calculate_precision_recall_at_k(
        model: NearestNeighbors,
        user_item_matrix: csr_matrix,
        test: pd.DataFrame,
        user_map: dict,
        item_map: dict,
        k: int = 10,
        threshold: float = 4.0
) -> tuple:
    """
    Calculate Precision@K and Recall@K.
    
    Args:
        model: trained model
        user_item_matrix: sparse matrix
        test: test dataframe
        user_map: user_id to index mapping
        item_map: movie_id to index mapping
        k: number of recommendations
        threshold: minimum rating to consider as liked
    Returns:
        tuple: precision@k, recall@k
    """
    precisions = []
    recalls = []

    for user_id, user_test in test.groupby("user_id"):
        user_idx = user_map.get(user_id)
        if user_idx is None:
            continue

        # movies user actually liked in test set
        liked_movies = set(
            user_test[user_test["rating"] >= threshold]["movie_id"].values
        )

        if len(liked_movies) == 0:
            continue

        # get user ratings from train matrix
        user_ratings = user_item_matrix[user_idx].toarray().flatten()
        unwatched = np.where(user_ratings == 0)[0]

        # predict scores for unwatched movies
        scores = []
        for item_idx in unwatched:
            movie_id = item_map.get(item_idx)
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

            scores.append((movie_id, pred))

        # get top K recommendations
        top_k = [mid for mid, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:k]]

        # calculate precision and recall
        hits = len(set(top_k) & liked_movies)
        precisions.append(hits / k)
        recalls.append(hits / len(liked_movies))

    precision = np.mean(precisions)
    recall = np.mean(recalls)

    logger.info(f"Precision@{k}: {precision:.4f}")
    logger.info(f"Recall@{k}: {recall:.4f}")

    return precision, recall

# full evaluation pipeline
def evaluate_pipeline(
        model: NearestNeighbors,
        user_item_matrix: csr_matrix,
        test: pd.DataFrame,
        user_map: dict,
        item_map: dict,
        k: int = 10
) -> dict:
    """
    Run full evaluation pipeline on best model.
    
    Args:
        model: best trained model
        user_item_matrix: sparse matrix
        test: test dataframe
        user_map: user_id to index mapping
        item_map: movie_id to index mapping
        k: number of recommendations
    Returns:
        dict: evaluation metrics
    """
    logger.info("Starting full evaluation pipeline...")

    # calculate metrics
    rmse = calculate_rmse(model, user_item_matrix, test, user_map, item_map)
    precision, recall = calculate_precision_recall_at_k(
        model, user_item_matrix, test, user_map, item_map, k=k
    )

    # compile report
    report = {
        "rmse": round(rmse, 4),
        f"precision@{k}": round(precision, 4),
        f"recall@{k}": round(recall, 4)
    }

    # save report
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_PATH / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    # log to mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    with mlflow.start_run(run_name="best_model_evaluation"):
        mlflow.log_metrics(report)

    logger.info(f"Evaluation report: {report}")
    logger.info(f"Report saved to {report_path} successfully!") 

    return report
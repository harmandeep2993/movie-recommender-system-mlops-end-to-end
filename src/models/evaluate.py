"""
Evaluation module for movie recommendation system.

Responsibilities:
- Evaluate best model on test set
- Calculate RMSE
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
from sklearn.metrics import root_mean_squared_error

from src.utils import get_logger

logger = get_logger(__name__)

REPORTS_PATH = Path(__file__).parent.parent.parent / "reports"


def save_evaluation_report(report: dict, report_name: str = "evaluation_report.json") -> None:
    """Save evaluation report to reports/ folder."""
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_PATH / report_name
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Evaluation report saved to {report_path} ✅")


def calculate_rmse(
    model: NearestNeighbors,
    user_item_matrix: csr_matrix,
    test: pd.DataFrame,
    user_map: dict,
    item_map: dict
) -> float:
    """
    Calculate RMSE on test sample.

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

    test_sample = test.sample(5000, random_state=42)

    for _, row in test_sample.iterrows():
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
    logger.info(f"RMSE on test sample: {rmse:.4f}")
    return rmse


def evaluate_pipeline(
    model: NearestNeighbors,
    user_item_matrix: csr_matrix,
    test: pd.DataFrame,
    user_map: dict,
    item_map: dict,
    k: int = 10
) -> dict:
    """
    Run evaluation pipeline on best model.

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
    logger.info("Starting evaluation pipeline...")
    rmse = calculate_rmse(model, user_item_matrix, test, user_map, item_map)

    report = {"rmse": round(rmse, 4)}

    save_evaluation_report(report)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    with mlflow.start_run(run_name="best_model_evaluation"):
        mlflow.log_metrics(report)

    logger.info(f"Evaluation report: {report}")

    return report
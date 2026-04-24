from src.data import load_dataset
from src.data import preprocess_pipeline
from src.features import build_features_pipeline
from src.models import train_pipeline, evaluate_pipeline, predict_pipeline
from sklearn.neighbors import NearestNeighbors

# step 1 - load
datasets = load_dataset()

# step 2 - preprocess
ratings_train, ratings_test, movies, users = preprocess_pipeline(
    datasets["ratings"],
    datasets["movies"],
    datasets["users"]
)

# step 3 - build features
user_item_matrix, user_map, item_map = build_features_pipeline(ratings_train)
print("Data loading, preprocessing, and feature engineering completed.")

# Step 4 - train model
best_model = train_pipeline(user_item_matrix, test=ratings_test, user_map=user_map, item_map=item_map, k_values=[10, 20, 50])
print("Model training completed.")

# step 5 - evaluate best model
evaluate_pipeline(best_model, user_item_matrix, ratings_test, user_map, item_map, k=10)
print("Evaluation completed.")

# step 6 - generate recommendations for a specific user

# user_id = 1
# recommendations = predict_pipeline(user_id, movies, n=10)
recommendations = predict_pipeline(user_id=1, movies=movies, n=10)
print(f"Top 10 recommendations for User 1:\n{recommendations}")
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Load data function
def load_data(user_item_path, spotify_path):
    user = pd.read_csv(user_item_path, low_memory=False)
    spotify = pd.read_csv(spotify_path, low_memory=False)
    return user, spotify

# Create item-based rating matrix function
def create_item_ratings_matrix(user_data):
    ratings_matrix = user_data.pivot(index='user_id', columns='id', values='listen_num').fillna(0)
    return ratings_matrix.T

# Split data into training and test sets function
def split_data(user_data, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(user_data, test_size=test_size, random_state=random_state)
    return train_data, test_data

# Create item-based rating matrices for training and test data function
def create_train_test_matrices(train_data, test_data):
    train_ratings_matrix = train_data.pivot(index='user_id', columns='id', values='listen_num').fillna(0).T
    test_ratings_matrix = test_data.pivot(index='user_id', columns='id', values='listen_num').fillna(0).T
    return train_ratings_matrix, test_ratings_matrix

# Create item similarity matrix function
def create_item_similarity_matrix(item_ratings_matrix):
    item_similarity_matrix = cosine_similarity(item_ratings_matrix)
    return pd.DataFrame(item_similarity_matrix, index=item_ratings_matrix.index, columns=item_ratings_matrix.index)

# Predict ratings function
def predict_ratings(similarity, ratings):
    ratings_diff = ratings - ratings.mean(axis=1)[:, np.newaxis]
    pred = similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    pred += ratings.mean(axis=1)[:, np.newaxis]
    return pred

# Calculate RMSE function
def calculate_rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return np.sqrt(((prediction - ground_truth) ** 2).mean())

# Main function
def main(user_item_path, spotify_path):
    # Load data
    user_data, _ = load_data(user_item_path, spotify_path)
    
    # Create item-based rating matrix
    item_ratings_matrix = create_item_ratings_matrix(user_data)
    
    # Split data into training and test sets
    train_data, test_data = split_data(user_data)
    
    # Create item-based rating matrices for training and test data
    train_ratings_matrix, test_ratings_matrix = create_train_test_matrices(train_data, test_data)
    
    # Create item similarity matrix
    item_similarity_df = create_item_similarity_matrix(item_ratings_matrix)
    
    # Predict ratings for the training data
    train_item_predicted_ratings = predict_ratings(item_similarity_df.values, train_ratings_matrix.values)
    
    # Predict ratings for the test data (only for users with actual ratings)
    test_item_predicted_ratings = predict_ratings(item_similarity_df.values, test_ratings_matrix.values)
    
    # Calculate RMSE for the test data
    test_rmse = calculate_rmse(test_item_predicted_ratings, test_ratings_matrix.values)
    print(f'Test RMSE: {test_rmse}')

if __name__ == "__main__":
    user_item_path = 'user_sample.csv'
    spotify_path = 'spotify.csv'
    main(user_item_path, spotify_path)

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def item_based_collaborative_filtering(user_data_path, spotify_data_path, test_size=0.2, random_state=42):
    # Load user and Spotify data
    user = pd.read_csv(user_data_path, low_memory=False)
    spotify = pd.read_csv(spotify_data_path, low_memory=False)

    # Create a ratings matrix with users as rows and items (songs) as columns
    ratings_matrix = user.pivot(index='user_id', columns='id', values='listen_num').fillna(0)

    # Calculate item similarity matrix using cosine similarity
    item_similarity_matrix = cosine_similarity(ratings_matrix.T)  # Transpose to get item-based similarity

    # Function to predict ratings using item-based collaborative filtering
    def predict_ratings_item_based(similarity, ratings):
        # Initialize an empty array for predicted ratings
        pred = np.zeros_like(ratings)

        # Loop through each item
        for i in range(len(ratings.T)):
            # Get the similarity vector for the current item
            item_similarity = similarity[i]

            # Calculate the weighted sum of ratings for the user
            weighted_sum = np.dot(item_similarity, ratings.T)
            weighted_sum_denominator = np.dot(item_similarity, (ratings.T != 0))

            # Avoid division by zero and set predicted ratings
            with np.errstate(divide='ignore', invalid='ignore'):
                pred[:, i] = np.where(weighted_sum_denominator != 0, weighted_sum / weighted_sum_denominator, 0)

        return pred

    # Split data into training and test sets
    train_data, test_data = train_test_split(user, test_size=test_size, random_state=random_state)

    # Create a training ratings matrix
    train_ratings_matrix = train_data.pivot(index='user_id', columns='id', values='listen_num').fillna(0)

    # Predict ratings using item-based collaborative filtering
    test_user_ids = test_data['user_id'].values
    test_item_ids = test_data['id'].values
    test_ratings = np.zeros(len(test_user_ids))

    for i in range(len(test_user_ids)):
        user_id = test_user_ids[i]
        item_id = test_item_ids[i]

        if item_id in train_ratings_matrix.columns:
            user_idx = train_ratings_matrix.index.get_loc(user_id)
            item_idx = train_ratings_matrix.columns.get_loc(item_id)

            test_ratings[i] = train_ratings_matrix.iloc[user_idx, item_idx]

    test_pred_ratings = predict_ratings_item_based(item_similarity_matrix, train_ratings_matrix.values)

    # Calculate RMSE for the test set
    test_rmse = np.sqrt(((test_pred_ratings - test_ratings) ** 2).mean())
    
    return test_rmse


rmse = item_based_collaborative_filtering('spotify_user.csv', 'spotify.csv')
print(f'Test RMSE: {rmse}')

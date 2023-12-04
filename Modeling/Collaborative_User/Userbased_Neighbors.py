import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import winsound

# Load data (This code assumes access to actual data files, but here it's provided as an example)
user = pd.read_csv('spotify_user.csv', low_memory=False)
spotify = pd.read_csv('spotify.csv', low_memory=False)

# Function to predict ratings given similarity matrix and ratings matrix
def predict_ratings(similarity, ratings, num_neighbors=10):
    """
    Predict ratings based on item-based collaborative filtering.
    
    Args:
    similarity (numpy.ndarray): Item similarity matrix.
    ratings (numpy.ndarray): Ratings matrix.
    num_neighbors (int): Number of neighbors to consider when making predictions.

    Returns:
    numpy.ndarray: Predicted ratings matrix.
    """
    # Calculate the mean rating for each user
    mean_user_rating = ratings.mean(axis=1)
    
    # Initialize an empty array to store predicted ratings
    pred = np.zeros_like(ratings)
    
    # Iterate over each user
    for i in range(len(ratings)):
        # Get the similarity vector for this user
        user_similarity = similarity[i]
        
        # Select the top num_neighbors neighbors based on similarity
        top_neighbors_indices = np.argsort(user_similarity)[::-1][:num_neighbors]
        
        # Calculate the weighted average of the neighbors' ratings
        for j in range(len(ratings[i])):
            if ratings[i][j] == 0:  # Predict only for items not yet rated by the user
                # Calculate the sum of weighted ratings of neighbors for this item
                weighted_sum = np.sum(user_similarity[top_neighbors_indices] * ratings[top_neighbors_indices, j])
                # Calculate the sum of absolute weights of neighbors
                weighted_sum_denominator = np.sum(np.abs(user_similarity[top_neighbors_indices]))
                
                # Calculate predicted rating using weighted average
                if weighted_sum_denominator > 0:
                    pred[i][j] = mean_user_rating[i] + weighted_sum / weighted_sum_denominator
                else:
                    pred[i][j] = 0  # Predict 0 if the denominator is 0 (to avoid division by zero)
    
    # Return the predicted ratings matrix
    return pred

# Split data into training and test sets
def split_data(user_data, test_size=0.2, random_state=42):
    """
    Split user data into training and test sets.
    
    Args:
    user_data (pandas.DataFrame): User data containing user-item ratings.
    test_size (float): Proportion of data to use as the test set.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    pandas.DataFrame: Training data.
    pandas.DataFrame: Test data.
    """
    train_data, test_data = train_test_split(user_data, test_size=test_size, random_state=random_state)
    return train_data, test_data

# Calculate RMSE for the test set
def calculate_rmse(true_ratings, pred_ratings):
    """
    Calculate Root Mean Squared Error (RMSE) between true and predicted ratings.

    Args:
    true_ratings (numpy.ndarray): True ratings.
    pred_ratings (numpy.ndarray): Predicted ratings.

    Returns:
    float: RMSE value.
    """
    # Filter out items with no true ratings
    mask = np.nonzero(true_ratings)
    pred_ratings = pred_ratings[mask]
    true_ratings = true_ratings[mask]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((true_ratings - pred_ratings) ** 2))
    return rmse

# Load data and create similarity matrix and ratings matrix
def load_and_prepare_data(user_file, spotify_file):
    """
    Load user and Spotify data, and prepare similarity and ratings matrices.

    Args:
    user_file (str): Path to the user data file.
    spotify_file (str): Path to the Spotify data file.

    Returns:
    pandas.DataFrame: User data.
    pandas.DataFrame: Spotify data.
    numpy.ndarray: Ratings matrix.
    numpy.ndarray: Similarity matrix.
    """
    # Load user and Spotify data
    user = pd.read_csv(user_file, low_memory=False)
    spotify = pd.read_csv(spotify_file, low_memory=False)

    # Create a ratings matrix with users as rows and items (songs) as columns
    ratings_matrix = user.pivot(index='user_id', columns='id', values='listen_num').fillna(0)

    # Calculate item similarity matrix using cosine similarity
    item_similarity_matrix = cosine_similarity(ratings_matrix.T)  # Transpose to get item-based similarity

    return user, spotify, ratings_matrix.values, item_similarity_matrix

# Main function to run item-based collaborative filtering
def main():
    # Load and prepare data
    user_data, spotify_data, ratings_matrix, item_similarity_matrix = load_and_prepare_data('spotify_user.csv', 'spotify.csv')

    # Split data into training and test sets
    train_data, test_data = split_data(user_data, test_size=0.2, random_state=42)

    # Predict ratings using item-based collaborative filtering
    test_user_ids = test_data['user_id'].values
    test_item_ids = test_data['id'].values
    test_ratings = np.zeros(len(test_user_ids))

    for i in range(len(test_user_ids)):
        user_id = test_user_ids[i]
        item_id = test_item_ids[i]

        if item_id in ratings_matrix.columns:
            user_idx = np.where(user_data['user_id'] == user_id)[0][0]
            item_idx = np.where(user_data.columns == item_id)[0][0]

            test_ratings[i] = ratings_matrix[user_idx, item_idx]

    test_pred_ratings = predict_ratings(item_similarity_matrix, ratings_matrix)

    # Calculate RMSE for the test set
    test_rmse = calculate_rmse(test_ratings, test_pred_ratings)
    print(f'Test RMSE: {test_rmse}')

    # Plot RMSE with increasing k (number of neighbors)
    rmses = []
    for k in range(1, 11):
        pred_ratings = predict_ratings(item_similarity_matrix, ratings_matrix, num_neighbors=k)
        rmse = calculate_rmse(test_ratings, pred_ratings)
        rmses.append(rmse)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), rmses, marker='o', linestyle='-', color='blue')
    plt.title('RMSE with Increasing k (Number of Neighbors)')
    plt.xlabel('Number of Neighbors: k')
    plt.ylabel('RMSE')
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show()

    # Beep sound to indicate completion
    frequency = 1000  # Sound frequency (Hz)
    duration = 5000  # Sound duration (milliseconds), changed to 5 seconds

    winsound.Beep(frequency, duration)

if __name__ == "__main__":
    main()

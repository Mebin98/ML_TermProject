import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(user_data_file, spotify_data_file):
    """
    Load user and Spotify data, and preprocess the user data.

    Args:
    user_data_file (str): File path of the user data CSV file.
    spotify_data_file (str): File path of the Spotify data CSV file.

    Returns:
    pandas.DataFrame: Preprocessed user data.
    """
    # Load user and Spotify data
    user = pd.read_csv(user_data_file, low_memory=False)
    spotify = pd.read_csv(spotify_data_file, low_memory=False)

    # Group by user_id and cluster to get the count of clusters per user
    cluster_counts_per_user = user.groupby(['user_id', 'cluster']).size().reset_index(name='count')

    # Determine the most frequent cluster per user
    most_frequent_cluster_per_user = cluster_counts_per_user.sort_values('count', ascending=False).drop_duplicates('user_id')

    # Find the most common cluster among all users
    most_common_cluster = most_frequent_cluster_per_user['cluster'].mode()[0]

    # Filter out users whose most frequent cluster is not the most common
    users_to_include = most_frequent_cluster_per_user[most_frequent_cluster_per_user['cluster'] == most_common_cluster]

    # Filter the original dataframe based on selected users
    user_df = user[user['user_id'].isin(users_to_include['user_id'])]

    # Filter users belonging to a specific cluster (e.g., cluster 0)
    user_df = user_df[user_df['cluster'] == 0]
    user_df.reset_index(drop=True, inplace=True)

    return user_df

def create_ratings_matrix(user_data):
    """
    Create the user-item ratings matrix.

    Args:
    user_data (pandas.DataFrame): Preprocessed user data containing user-item ratings.

    Returns:
    pandas.DataFrame: User-item ratings matrix.
    """
    ratings_matrix = user_data.pivot(index='user_id', columns='id', values='listen_num').fillna(0)
    return ratings_matrix

def calculate_similarity_matrix(ratings_matrix):
    """
    Calculate the item similarity matrix using cosine similarity.

    Args:
    ratings_matrix (pandas.DataFrame): User-item ratings matrix.

    Returns:
    pandas.DataFrame: Item similarity matrix.
    """
    similarity_matrix = cosine_similarity(ratings_matrix.T)  # Transpose to get item-based similarity
    similarity_df = pd.DataFrame(similarity_matrix, index=ratings_matrix.columns, columns=ratings_matrix.columns)
    return similarity_df

def split_train_test_data(user_data, test_size=0.2, random_state=42):
    """
    Split the user data into training and test sets.

    Args:
    user_data (pandas.DataFrame): Preprocessed user data.
    test_size (float): Proportion of data to use as the test set.
    random_state (int): Random seed for reproducibility.

    Returns:
    pandas.DataFrame: Training data.
    pandas.DataFrame: Test data.
    """
    train_data, test_data = train_test_split(user_data, test_size=test_size, random_state=random_state)
    return train_data, test_data

def predict_ratings(similarity_matrix, ratings_matrix):
    """
    Predict user ratings based on item similarity and ratings matrix.

    Args:
    similarity_matrix (pandas.DataFrame): Item similarity matrix.
    ratings_matrix (pandas.DataFrame): User-item ratings matrix.

    Returns:
    pandas.DataFrame: Predicted ratings matrix.
    """
    mean_user_rating = ratings_matrix.mean(axis=1)
    ratings_diff = (ratings_matrix - mean_user_rating[:, np.newaxis])
    pred = mean_user_rating[:, np.newaxis] + similarity_matrix.dot(ratings_diff) / np.array([np.abs(similarity_matrix).sum(axis=1)]).T
    return pred

def calculate_rmse(true_ratings, pred_ratings):
    """
    Calculate Root Mean Squared Error (RMSE) between true and predicted ratings.

    Args:
    true_ratings (numpy.ndarray): True ratings.
    pred_ratings (numpy.ndarray): Predicted ratings.

    Returns:
    float: RMSE value.
    """
    mask = np.nonzero(true_ratings)
    pred_ratings = pred_ratings[mask]
    true_ratings = true_ratings[mask]
    rmse = np.sqrt(((pred_ratings - true_ratings) ** 2).mean())
    return rmse

def main(user_data_file, spotify_data_file):
    # Load and preprocess data
    user_data = load_and_preprocess_data(user_data_file, spotify_data_file)

    # Create ratings matrix
    ratings_matrix = create_ratings_matrix(user_data)

    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(ratings_matrix)

    # Split data into training and test sets
    train_data, test_data = split_train_test_data(user_data)

    # Predict ratings using item-based collaborative filtering
    train_pred_ratings = predict_ratings(similarity_matrix, ratings_matrix)
    test_pred_ratings = predict_ratings(similarity_matrix, ratings_matrix)

    # Calculate RMSE for the test set
    test_rmse = calculate_rmse(test_data['listen_num'].values, test_pred_ratings[test_data.index, :])
    print(f'Test RMSE: {test_rmse}')

if __name__ == "__main__":
    user_data_file = 'spotify_user.csv'  # Replace with the path to your user data file
    spotify_data_file = 'spotify.csv'  # Replace with the path to your Spotify data file
    main(user_data_file, spotify_data_file)



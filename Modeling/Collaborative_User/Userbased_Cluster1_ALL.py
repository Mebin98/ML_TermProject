import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Function to load and preprocess user data
def load_and_preprocess_data(user_data_path, most_common_cluster):
    user = pd.read_csv(user_data_path, low_memory=False)
    
    # Group by user_id and cluster to get the count of clusters per user
    cluster_counts_per_user = user.groupby(['user_id', 'cluster']).size().reset_index(name='count')
    
    # Determine the most frequent cluster per user
    most_frequent_cluster_per_user = cluster_counts_per_user.sort_values('count', ascending=False).drop_duplicates('user_id')
    
    # Filter out users whose most frequent cluster is not the most common
    users_to_include = most_frequent_cluster_per_user[most_frequent_cluster_per_user['cluster'] == most_common_cluster]
    
    # Filter the original dataframe based on selected users
    user_df = user[user['user_id'].isin(users_to_include['user_id'])]
    
    return user_df

# Function to create user-item ratings matrix and similarity matrix
def create_ratings_and_similarity_matrices(user_df):
    # Create a user-item ratings matrix
    ratings_matrix = user_df.pivot(index='user_id', columns='id', values='listen_num').fillna(0)

    # Calculate cosine similarity between items
    similarity_matrix = cosine_similarity(ratings_matrix)

    # Convert the similarity matrix to a DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=ratings_matrix.index, columns=ratings_matrix.index)
    
    return ratings_matrix, similarity_df

# Function to split data into training and test sets
def split_data(user_df, test_size=0.2, random_state=42):
    # Split the data into training and test sets
    train_data, test_data = train_test_split(user_df, test_size=test_size, random_state=random_state)
    
    # Create user-item ratings matrices for training and testing
    train_ratings_matrix = train_data.pivot(index='user_id', columns='id', values='listen_num').fillna(0)
    test_ratings_matrix = test_data.pivot(index='user_id', columns='id', values='listen_num').fillna(0)
    
    return train_ratings_matrix, test_ratings_matrix

# Function to predict ratings
def predict_ratings(similarity, ratings):
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred

# Function to calculate RMSE
def calculate_rmse(predicted_ratings, actual_ratings):
    predicted_ratings = predicted_ratings[actual_ratings.nonzero()].flatten()
    actual_ratings = actual_ratings[actual_ratings.nonzero()].flatten()
    rmse = np.sqrt(((predicted_ratings - actual_ratings) ** 2).mean())
    return rmse

# Main function to perform recommendation and evaluation
def main(user_data_path, most_common_cluster, test_size=0.2, random_state=42):
    # Load and preprocess data
    user_df = load_and_preprocess_data(user_data_path, most_common_cluster)
    
    # Create user-item ratings matrix and similarity matrix
    ratings_matrix, similarity_df = create_ratings_and_similarity_matrices(user_df)
    
    # Split data into training and test sets
    train_ratings_matrix, test_ratings_matrix = split_data(user_df, test_size, random_state)
    
    # Predict ratings for the training set
    train_user_predicted_ratings = predict_ratings(similarity_df.values, train_ratings_matrix.values)
    
    # Predict ratings for the test set (only for users with actual ratings)
    test_user_predicted_ratings = predict_ratings(similarity_df.values, test_ratings_matrix.values)
    
    # Calculate RMSE for the test set
    test_rmse = calculate_rmse(test_user_predicted_ratings, test_ratings_matrix.values)
    
    print(f'Test RMSE: {test_rmse}')

if __name__ == "__main__":
    # Specify the path to the user data CSV and the most common cluster
    user_data_path = 'spotify_user.csv'
    most_common_cluster = 1
    
    # Call the main function to perform recommendation and evaluation
    main(user_data_path, most_common_cluster)

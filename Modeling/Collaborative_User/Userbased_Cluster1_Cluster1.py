import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_path, most_common_cluster):
    # Load user data
    user = pd.read_csv(data_path, low_memory=False)

    # Group by user_id and cluster to count clusters per user
    cluster_counts_per_user = user.groupby(['user_id', 'cluster']).size().reset_index(name='count')

    # Determine the most frequent cluster per user
    most_frequent_cluster_per_user = cluster_counts_per_user.sort_values('count', ascending=False).drop_duplicates('user_id')

    # Filter users whose most frequent cluster is the most common
    users_to_include = most_frequent_cluster_per_user[most_frequent_cluster_per_user['cluster'] == most_common_cluster]

    # Filter the original dataframe based on selected users
    user_df = user[user['user_id'].isin(users_to_include['user_id'])]

    return user_df

def create_ratings_and_similarity_matrices(user_df):
    # Create user-item ratings matrix
    ratings_matrix = user_df.pivot(index='user_id', columns='id', values='listen_num').fillna(0)

    # Calculate cosine similarity between items
    similarity_matrix = cosine_similarity(ratings_matrix)

    # Convert the similarity matrix to a DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=ratings_matrix.index, columns=ratings_matrix.index)

    return ratings_matrix, similarity_df

def split_data(user_df, test_size=0.2, random_state=42):
    # Split data into training and test sets
    train_data, test_data = train_test_split(user_df, test_size=test_size, random_state=random_state)

    # Create user-item ratings matrices for training and testing
    train_ratings_matrix = train_data.pivot(index='user_id', columns='id', values='listen_num').fillna(0)
    test_ratings_matrix = test_data.pivot(index='user_id', columns='id', values='listen_num').fillna(0)

    return train_ratings_matrix, test_ratings_matrix

def predict_ratings(similarity, ratings):
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred

def calculate_rmse(predicted_ratings, actual_ratings):
    predicted_ratings = predicted_ratings[actual_ratings.nonzero()].flatten()
    actual_ratings = actual_ratings[actual_ratings.nonzero()].flatten()
    rmse = np.sqrt(((predicted_ratings - actual_ratings) ** 2).mean())
    return rmse

def main(data_path, most_common_cluster, test_size=0.2, random_state=42):
    user_df = load_and_preprocess_data(data_path, most_common_cluster)
    ratings_matrix, similarity_df = create_ratings_and_similarity_matrices(user_df)
    train_ratings_matrix, test_ratings_matrix = split_data(user_df, test_size, random_state)
    train_user_predicted_ratings = predict_ratings(similarity_df.values, train_ratings_matrix.values)
    test_user_predicted_ratings = predict_ratings(similarity_df.values, test_ratings_matrix.values)
    test_rmse = calculate_rmse(test_user_predicted_ratings, test_ratings_matrix.values)
    print(f'Test RMSE: {test_rmse}')

if __name__ == "__main__":
    data_path = 'spotify_user.csv'
    most_common_cluster = 1
    main(data_path, most_common_cluster)

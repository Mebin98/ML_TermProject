import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Read Spotify data from CSV file
spotify_data = pd.read_csv('C:\\Users\\gus_0\\Desktop\\spotify_data.csv', low_memory=False)

# Drop the 'explicit' column
spotify_data = spotify_data.drop('explicit', axis=1)

# Convert the 'popularity' column to numeric and handle NaN values
spotify_data['popularity'] = pd.to_numeric(spotify_data['popularity'], errors='coerce')
spotify_data = spotify_data.dropna()

# List of numeric columns for feature scaling
number_cols = [
    'valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
    'loudness', 'mode', 'popularity', 'speechiness', 'tempo'
]

def get_song_data(name, year, spotify_data):
    try:
        # Find the song in the dataframe
        song_data = spotify_data[(spotify_data['name'] == name) & (spotify_data['year'] == year)].iloc[0]
        return song_data
    except IndexError:
        # Return None if the song is not found
        return None

def flatten_dict_list(dict_list):
    flattened_dict = {}
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song['name'], song['year'], spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in the database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].astype(float).values
        song_vectors.append(song_vector)
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def recommend_songs_in_cluster(song_list, spotify_data, n_songs=11):
    metadata_cols = ['id', 'name', 'year', 'artists', 'cluster']
    song_dict = flatten_dict_list(song_list)
    
    # Calculate the mean vector for the input songs
    song_center = get_mean_vector(song_list, spotify_data)
    
    # Get the cluster value for the input song
    input_song_cluster = get_song_data(song_list[0]['name'], song_list[0]['year'], spotify_data)['cluster']
    
    # Find songs in the same cluster
    cluster_data = spotify_data[spotify_data['cluster'] == input_song_cluster]
    scaled_data = cluster_data[number_cols]  # Use only numeric columns
    scaled_cluster_center = pd.DataFrame([song_center], columns=number_cols)
    
    # Calculate cosine similarity
    distances = cdist(scaled_cluster_center, scaled_data, 'cosine')
    
    # Get indices of recommended songs
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = cluster_data.iloc[index]
    rec_songs = rec_songs[~((rec_songs['name'].isin(song_dict['name'])) & (rec_songs['year'].isin(song_dict['year'])))]
    
    return rec_songs[metadata_cols].to_dict(orient='records')

# Example data
song_list = [{'name': 'Steal The Show', 'year': 2023}]

# Get recommended songs using the cluster-based recommendation algorithm
recommended_songs = recommend_songs_in_cluster(song_list, spotify_data)

# Print the recommended songs
print("Recommended Songs:")
print()
for song in recommended_songs:
    print(f"cluster : {song['cluster']}")
    print(f"Song ID : {song['id']}")
    print(f"Name: {song['name']}")
    print(f"Year: {song['year']}")
    print(f"Artists: {song['artists']}")
    print("="*30)  # Separator

# Evaluation data with actual ratings
evaluation_data = [
    {'music_id': '31bsuKDOzFGzBAoXxtnAJm', 'rating': 5},
    {'music_id': '51pQ7vY7WXzxskwloaeqyj', 'rating': 3},
    {'music_id': '2JltnZUDzrxKuwNjkK5N6Q', 'rating': 4},
    {'music_id': '65YDMuJmyF8cxTrk4Xogy0', 'rating': 4},
    {'music_id': '3PFaFVWq5wucLu6s4baj9D', 'rating': 4},
    {'music_id': '3hEkAosl2ZuO9FltxY5L1R', 'rating': 4},
    {'music_id': '7bG6SQBGZthPDG5QJL5Gf7', 'rating': 2},
    {'music_id': '1J3w85cS3FEmoSKRu2dQJ8', 'rating': 6},
    {'music_id': '3gruCKVuhCsRjs3a3dG8CA', 'rating': 7},
    {'music_id': '3KPwt1LBpt1jVSHz8GXERo', 'rating': 8},
]

# Dictionary to store predicted ratings
predicted_ratings = {}

# Apply the recommendation algorithm and store predicted ratings
for entry in evaluation_data:
    music_id = entry['music_id']
    
    # Apply the recommendation algorithm for the specific music and calculate the predicted rating
    recommended_songs = recommend_songs_in_cluster([{'name': 'Steal The Show', 'year': 2023}], spotify_data)
    
    # For now, unify predicted ratings to 9
    predicted_rating = 9
    
    # Store the predicted result
    predicted_ratings[(music_id)] = predicted_rating

# Create a DataFrame for predicted ratings
predicted_data = pd.DataFrame([
    {'music_id': music_id, 'predicted_rating': predicted_ratings.get((music_id), np.nan)}
    for music_id in predicted_ratings.keys()
])

# Merge evaluation and predicted dataframes
evaluation_data_df = pd.DataFrame(evaluation_data)
evaluation_data_df = pd.merge(evaluation_data_df, predicted_data, on=['music_id'])

# Compute RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(evaluation_data_df['rating'], evaluation_data_df['predicted_rating']))
print(f"RMSE : {rmse}")

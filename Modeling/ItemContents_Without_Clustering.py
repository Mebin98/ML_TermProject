import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist

# Load Spotify data from CSV file
spotify_data = pd.read_csv('C:\\Users\\gus_0\\Desktop\\spotify_data.csv', low_memory=False)

# Drop the 'explicit' column
spotify_data = spotify_data.drop('explicit', axis=1)

# Convert the 'popularity' column to numeric type; handle errors by coercing to NaN and drop NaN values
spotify_data['popularity'] = pd.to_numeric(spotify_data['popularity'], errors='coerce')
spotify_data = spotify_data.dropna()

# List of numeric columns to be used in the analysis
number_cols = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
               'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Function to retrieve song data based on name and year
def get_song_data(name, year, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == name) & (spotify_data['year'] == year)].iloc[0]
        return song_data
    except IndexError:
        return None

# Function to flatten a list of dictionaries
def flatten_dict_list(dict_list):
    flattened_dict = {}
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

# Function to calculate the mean vector of a list of songs
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

# Function to recommend songs based on a list of input songs
def recommend_songs(song_list, spotify_data, n_songs=11):
    metadata_cols = ['id', 'name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    # Calculate the mean vector for the input songs
    song_center = get_mean_vector(song_list, spotify_data)
    scaled_data = spotify_data[number_cols]  # Extract only numeric data
    scaled_song_center = pd.DataFrame([song_center], columns=number_cols)
    
    # Calculate cosine similarities
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    
    # Extract indices to return the corresponding songs
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    # Exclude songs already in the input list
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    
    return rec_songs[metadata_cols].to_dict(orient='records')

# Evaluation data for testing
evaluation_data = [
    {'music_id': '31bsuKDOzFGzBAoXxtnAJm', 'rating': 5},
    {'music_id': '51pQ7vY7WXzxskwloaeqyj', 'rating': 3},
    {'music_id': '2JltnZUDzrxKuwNjkK5N6Q', 'rating': 4},
    {'music_id': '5RxpYHVbGJPOvSEATQyg9P', 'rating': 3},
    {'music_id': '65YDMuJmyF8cxTrk4Xogy0', 'rating': 4},
    {'music_id': '4gxzgTzA4MSLf5rje1rX65', 'rating': 2},
    {'music_id': '1J3w85cS3FEmoSKRu2dQJ8', 'rating': 7},
    {'music_id': '4iddJAOsc6U0hJ3krSJAKn', 'rating': 7},
    {'music_id': '2U9kDk5mlHYunC7PvbZ8KX', 'rating': 9},
    {'music_id': '41o2ydrj7Xm9Yt5odIBqq4', 'rating': 6}
]

# Dictionary to store predicted ratings
predicted_ratings = {}

# Apply the recommendation algorithm and store predicted ratings
for entry in evaluation_data:
    music_id = entry['music_id']
    
    # Apply the recommendation algorithm for the specific music and calculate the predicted rating
    recommended_songs = recommend_songs([{'name': 'Steal The Show', 'year': 2023}], spotify_data)
    
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

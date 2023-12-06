from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np

# Load Spotify data from CSV file
spotify_data = pd.read_csv('spotify_data.csv', low_memory=False)

# Drop the 'explicit' column
spotify_data = spotify_data.drop('explicit', axis=1)

# Convert the 'popularity' column to numeric type; handle errors by coercing to NaN and drop NaN values
spotify_data['popularity'] = pd.to_numeric(spotify_data['popularity'], errors='coerce')
spotify_data = spotify_data.dropna()

# List of numeric columns to be used in the analysis
number_cols = [
    'valence', 'acousticness', 'danceability',
    'energy', 'instrumentalness', 'key', 'liveness',
    'loudness', 'mode', 'popularity', 'speechiness', 'tempo'
]

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

    if isinstance(dict_list, dict):
        for key, value in dict_list.items():
            flattened_dict[key] = value
    else:
        # dict_list가 리스트 형태로 들어오는 경우
        for key in dict_list[0].keys():
            flattened_dict[key] = dict_list[0][key]

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
    metadata_cols = ['id', 'name', 'year', 'artists', 'valence', 'acousticness','danceability', 'energy']

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
    #rec_songs = rec_songs[~rec_songs['name'].isin([song_dict['name']])]


    return rec_songs[metadata_cols].to_dict(orient='records')


song_list = [{'name': 'Steal The Show', 'year': 2023}]
recommend = recommend_songs(song_list, spotify_data, n_songs=11)
print(recommend)

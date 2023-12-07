import pandas as pd
import numpy as np
import random
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error

# Load Spotify data from CSV file
spotify_data = pd.read_csv('C:\\Users\\gus_0\\Desktop\\spotify_data.csv', low_memory=False)
spotify_data = spotify_data.drop('explicit', axis=1)
spotify_data['popularity'] = pd.to_numeric(spotify_data['popularity'], errors='coerce')
spotify_data = spotify_data.dropna()

# List of numeric columns for analysis
number_cols = [
    'valence', 'acousticness', 'danceability',
    'energy', 'instrumentalness', 'key', 'liveness',
    'loudness', 'mode', 'popularity', 'speechiness', 'tempo'
]

# Function to retrieve song data based on name and year
def get_song_data(name, year):
    song_data = spotify_data[(spotify_data['name'] == name) & (spotify_data['year'] == year)].iloc[0]
    return song_data

# Function to calculate the mean vector of a list of songs
def get_mean_vector(song_list):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song['name'], song['year'])
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in the database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].astype(float).values
        song_vectors.append(song_vector)
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

# Function to recommend songs based on a list of input songs
def recommend_songs(song_list, n_songs=11):
    song_center = get_mean_vector(song_list)
    scaled_data = spotify_data[number_cols]
    scaled_song_center = pd.DataFrame([song_center], columns=number_cols)
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]
    return rec_songs

# Create a random dataset for evaluation
def create_random_dataset(num_samples=100, random_seed=42):
    user_data = pd.read_csv('C:\\Users\\gus_0\\Desktop\\user.csv')
    random_sample = user_data.sample(n=num_samples, random_state=random_seed)
    random_sample['rating'] = random_sample['listen_num'] // 10
    evaluation_data = random_sample[['music_id', 'rating']]
    evaluation_data_list = evaluation_data.to_dict(orient='records')
    return evaluation_data_list

# Evaluation dataset
evaluation_data = create_random_dataset(num_samples=1000, random_seed=42)

# Dictionary to store predicted ratings
predicted_ratings = {}

# Apply the recommendation algorithm and store predicted ratings
for entry in evaluation_data:
    music_id = entry['music_id']
    recommended_songs = recommend_songs([{'name': 'Steal The Show', 'year': 2023}])
    for rec_song in recommended_songs:
        predicted_rating = random.randint(1, 10)
        predicted_ratings[(music_id, rec_song['id'])] = predicted_rating

# Create a DataFrame for predicted ratings
predicted_data = pd.DataFrame(list(predicted_ratings.items()), columns=['music_id', 'predicted_rating'])
predicted_data[['music_id', 'song_id']] = pd.DataFrame(predicted_data['music_id'].tolist(), index=predicted_data.index)

# Merge evaluation and predicted dataframes
evaluation_data_df = pd.DataFrame(evaluation_data)
evaluation_data_df = pd.merge(evaluation_data_df, predicted_data, on=['music_id', 'music_id'], how='left')

# Fill NaN values with a default value (you can adjust it as needed)
default_value = 0
evaluation_data_df['predicted_rating'].fillna(default_value, inplace=True)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(evaluation_data_df['rating'], evaluation_data_df['predicted_rating']))
print(f"RMSE : {rmse}")

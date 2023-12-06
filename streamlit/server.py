from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from starlette.middleware.cors import CORSMiddleware
import requests
from contentBased import recommend_songs, flatten_dict_list
import pandas as pd
import traceback

app = FastAPI()

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


# CORS 설정
origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/recommend/{song_name}/{song_year}")
async def recommend(song_name: str, song_year: int):

    try:
        song_list = [{'name': song_name, 'year': song_year}]
        print(song_list)

        recommend_result = recommend_songs(song_list, spotify_data, n_songs=11)
        print("Recommend_result :", recommend_result)
        # contentbased.py에 요청 보내기

        return {"recommendation": recommend_result}

    except Exception as e:
        # Log the exception traceback for debugging
        traceback.print_exc()

        # Handle the exception and return an error response
        error_message = f"An error occurred: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)

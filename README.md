# Gachon University [Machine Learning Term Project]
Director : Prof. Won Kim

Students : Hobin Hwang, Gahyun Kim, Sengwoo Han, Yongho Bae

# Subject 
Music Recommendation System

# Data
[Spotify Dataset](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset)

# Business Objective
1. Implement a recommendation system similar to those provided by acutal systems as possible.
2. Conducting efficient preprocessing to increase computing speed and avoid memory issue.
3. Not just considering ratings for music, but making recommendations by taking into account the diverse characteristics of users and the features of items.


# Data Exploration
(1) Feature Explanations

    Spotify dataset have 19 features as follows:

    valence : The brightness level of the track (higher values indicate a brighter and happier sound, while lower values suggest a sadder and more melancholic tone).

    year  : Release year
    
    acousticness : range 0~1, degree of closeness to pure sound.![image](https://github.com/Mebin98/ML_TermProject/assets/121173175/97349681-6c29-49c7-9b56-fea83ecd4706)

    Artists : Singer or composer

    danceability : range 0~1, the bigger, better for dancing!

    duration_ms : length of music

    energy : 0~1 range, Faster and more flamboyant, the value increases with more noise.

    explicit : 1 or 0, whether not good for children or else.

    id : primary key of music

    instrumentalness : degree of vocals in the song

    key : mapping using the standard pitch class notation

    liveness : the degree of liveness of the song, the value is higher for live recordings.

    loudness : the degree of brilliance/loudness of the song

    mode : major(0), minor(1)

    name : name of the music

    popularity : range 0~100

    release_date : release_date of music

    speechiness : degree of saying

    tempo : BPM

    cluster : cluster of music (will be allocated through clustering algorithm)

(2) User`s data

    Unfortunately, Spotify doesn`t provide user`s data. So we created user`s dataset by ourselves. 

    User`s data has 4 features as follows:

    user_id : primary key of user

    listen_num : number of times listened to this song

    Music_id : foregin key of user data(primary key of music data) 

    clsuter of music : cluster of music (will be allocated through clustering algorithm)

# Data Preprocessing
(1) Data update with Spotify Open API
[Spotify Open API](https://spotipy.readthedocs.io/en/2.16.0/)

(2) Data Filtering through Clustering

    <Scaling>
    MinMaxScaling
    StandardScaling
    Normalizer
    QuantileTransformer
    RobustScaling

    <Clustering>
    KMeans
    Clarans
    Spectral
    DBSCAN

<Clustering Result>
Spectral
<img width="848" alt="spectral_clustering" src="https://github.com/Mebin98/ML_TermProject/assets/121173175/1739d63c-2308-475d-8a05-16a9d8e81f37">
<img width="856" alt="spectral_evaluation" src="https://github.com/Mebin98/ML_TermProject/assets/121173175/b7cb7860-35f6-4d14-ba01-e0f0a646fb1a">

KMeans
<img width="1541" alt="KMeans_Cluster_1" src="https://github.com/Mebin98/ML_TermProject/assets/121173175/25b18d1d-e12a-43d2-88c3-58f2e3d7f62e">
![KMeans_Evaluation_1](https://github.com/Mebin98/ML_TermProject/assets/121173175/a6b0512c-afff-47b6-bcca-1c877fab3477)
    





    



    







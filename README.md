# Gachon University [Machine Learning Term Project]
Director : Prof. Won Kim

Developers : Hobin Hwang, Gahyun Kim, Sengwoo Han, Yongho Bae

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
**Clustering Result**

Spectral
<img width="869" alt="image" src="https://github.com/Mebin98/ML_TermProject/assets/121173175/445a7758-3a07-459c-bdf5-296afc3a4153">

KMeans
<img width="872" alt="image" src="https://github.com/Mebin98/ML_TermProject/assets/121173175/ad9038e4-a287-48ee-b9f1-caf105f2276d">

We selected the combination(KMeans, Normalizer, k=4) based on direction of clusters!

<img width="339" alt="normal" src="https://github.com/Mebin98/ML_TermProject/assets/121173175/81b8df10-8a6d-4e44-a71c-873772e89f98">



# Modeling 

Recommendation System Algorithm

    1. Collaborative Filtering(User-Based) <RMSE : 4.247~4.933>
    2. Collaborative Filtering(Item-Based) <RMSE : 6.028>
    3. Item content`s Based <RMSE : 4.180>
    
# Program Structure

```
C:.
|   README.md
|   spotify.csv
|
+---Clustering
|       Clarans_Clustering.py 
|       DBSCAN_Clustering.py
|       KMeans_Clustering.py
|       selected_normalized_data_k3.csv // KMeans, k=3, Normalizer
|       selected_normalized_data_k4.csv // KMeans, k=4, Noramlizer
|       selected_normalized_data_k5.csv // KMeans, k=4, Normalizer
|       Spectral_Clustering.py
|
+---Modeling
|   +---Collaborative_Item
|   |       ItemBased.py  // Item based Recommendation Code(Memory Error)
|   |       ItemBased_Sampling.py // Alternative code(Sampling)
|   |
|   +---Collaborative_User
|   |       Userbased_Cluster0_ALL.py // (Music data : Cluster0, User data : ALL)
|   |       Userbased_Cluster0_Cluster0.py // (Music data : Cluster0, User data : cluster 0 most)
|   |       Userbased_Cluster1_ALL.py // (Music data : Cluster1, User data : ALL)
|   |       Userbased_Cluster1_Cluster1.py // (Music data : Cluster1, User data : cluster 1 most)
|   |       Userbased_Neighbors.py 
|   |
|   \---ItemContents
|           ItemContents_Without_Clustering.py
|           ItemContents_With_Clustering.py
|           ItemContent_Revised.py // Final code of Item contents based
|
+---Preprocessing
|   |   Spotify_Preprocessing.py
|   |
|   \---csvFile
|           scaled_data_minmax.csv
|           scaled_data_normalizer.csv
|           scaled_data_quantile.csv
|           scaled_data_robust.csv
|           scaled_data_standard.csv
|
+---streamlit // Normal version for web
|       contentBased.py
|       main.py
|       server.py
|
\---streamlit_SageMaker // Web for SageMaker version
        contentBased.py
        main.py
        server.py
        storage_handler.py
```

# Installation(For normal version)

First, you must install the libraries as follows:

```
pip install spotipy
pip install scikit-learn
pip install pandas
pip install numpy
pip install streamlit
pip install requests
pip install matplotlib
pip install fastapi
pip install uvicorn
pip install starlette
```

Then, install 'streamlit' direcotory!

Also, you must install spotify dataset then it should be in streamlit direcotry.

Then, open terminal at admin mode.

```
uvicorn server:app --reload
```

Then, open an another terminal.

```
streamlit run main.py
```

You must make sure that all the files be in same direcotory!


# Youtube Demo(SageMaker Version)

[Youtube Demo](https://www.youtube.com/watch?v=j3ih_GLqEP4)





    



    







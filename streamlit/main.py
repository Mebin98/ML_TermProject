import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd

print("page reload")

initial_musics = [{
    "Song": "",
    "Artist": "",
    "Music_id": "",
    "Year": "",
    "valence": "",
    "acousticness": "",
    "danceability": "",
    "energy": ""
}]

# 페이지가 처음 로딩될 때 session_state를 초기화
if "musics" not in st.session_state:
    st.session_state.musics = initial_musics

st.title("Music Recommedation")

# <hr>
st.text("")
st.markdown("---")
st.text("")

# 사용자로부터 입력 받기
song_name = st.text_input("Enter the song name:")
song_year = st.text_input("Enter the song year:")

# "Recommend" 버튼 클릭 시 동작
if st.button("Recommend"):

    # <hr>
    st.text("")
    st.markdown("---")
    st.text("")
    
    # 서버에 요청 보내기
    response = requests.get(f"http://127.0.0.1:8000/api/recommend/{song_name}/{song_year}")

    # 서버의 응답 처리
    if response.status_code == 200:
        recommended_songs = response.json()["recommendation"]
        print("Recommended Musics : ", recommended_songs)
        print(response.json())

        # 결과 출력
        st.session_state.recommended_songs = recommended_songs
        for song in recommended_songs:
            # musics 리스트에 노래 정보 추가
            st.session_state.musics.append({
                "Song": song['name'],
                "Artist": song['artists'],
                "Music ID": song['id'],
                "Year": song['year'],
                "valence": song['valence'],
                "acousticness": song['acousticness'],
                "danceability": song['danceability'],
                "energy": song['energy']
            })
    

    else:
        st.error("Error getting recommendations. Please try again.")

#musics list의 index 1의 name, artist, year 정보 출력
if len(st.session_state.musics) > 1:
    selected_song = st.session_state.musics[1]
    st.write(f"{selected_song['Song']} by {selected_song['Artist']} ({selected_song['Year']})을 선택하셨습니다.")



# <hr>
st.text("")
st.markdown("---")
st.text("")

st.write("Recommended Songs :")

# 추천 음악 리스트 출력
for music in st.session_state.musics[2:]: # [0]은 empty, [1]은 자기자신 -> 2부터 출력해야 10개
    st.write(f"{music['Song']} by {music['Artist']} ({music['Year']})")



# Create a list of song titles for multi-selection
song_titles = [song['Song'] for song in st.session_state.musics]

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def set_clicked():
    st.session_state.clicked = True

# <hr>
st.text("")
st.markdown("---")
st.text("")


st.button('결과 분석', on_click=set_clicked)
if st.session_state.clicked:


    
    # Allow the user to select multiple songs
    input_songs = st.multiselect("Select songs for analysis:", song_titles)
    # <hr>
    st.text("")
    st.markdown("---")
    st.text("")
    
    # Filter DataFrame for selected songs
    selected_df = pd.DataFrame(st.session_state.musics[1:]).loc[
        pd.DataFrame(st.session_state.musics[1:])['Song'].isin(input_songs)
    ]

    # Check if selected_df is not empty
    if not selected_df.empty:
        # Create an empty dictionary to store plot data
        plot_data = {'x': ['valence', 'acousticness', 'danceability', 'energy'], 'labels': []}

        for index, row in selected_df.iterrows():
            label = f"{row['Song']} by {row['Artist']} ({row['Year']})"
            data = [row['valence'], row['acousticness'], row['danceability'], row['energy']]
            plot_data['labels'].append(label)

            if 'y' not in plot_data:
                plot_data['y'] = [data]
            else:
                plot_data['y'].append(data)

        # Plot features for each selected song
        fig, ax = plt.subplots(figsize=(10, 6))

        for label, y_data in zip(plot_data['labels'], plot_data['y']):
            ax.plot(plot_data['x'], y_data, marker='o', label=label)

        ax.set_xlabel("Features")
        ax.set_ylabel("Value")
        plt.legend(loc='lower right', bbox_to_anchor=(1.0,1.0))

        st.pyplot(fig, bbox_inches='tight', pad_inches=0)
    else:
        st.warning("No selected songs for analysis.")

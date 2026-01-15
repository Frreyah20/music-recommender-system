import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity

# PAGE CONFIG
st.set_page_config(page_title="Music Recommendation System", layout="wide")
st.title("Music Recommendation System")
st.write("Select a song and choose how you want similar songs to be recommended.")

#Load datset
#Loads everything once and then reuses it
@st.cache_data
def load_assets():
    with open("df_global.pkl", "rb") as f:
        df_global = pickle.load(f)
    with open("df_regional.pkl", "rb") as f:
        df_regional = pickle.load(f)

    with open("music_matrix_global.pkl", "rb") as f:
        music_global = pickle.load(f)
    with open("vibe_matrix_global.pkl", "rb") as f:
        vibe_global = pickle.load(f)

    with open("music_matrix_regional.pkl", "rb") as f:
        music_regional = pickle.load(f)
    with open("vibe_matrix_regional.pkl", "rb") as f:
        vibe_regional = pickle.load(f)

    return (
        df_global, df_regional,
        music_global, vibe_global,
        music_regional, vibe_regional
    )

(
    df_global, df_regional,
    music_global, vibe_global,
    music_regional, vibe_regional
) = load_assets()


# Build a unified dropdown
def build_dropdown(df):
    return (df["track_name"] + " - " + df["artist_name"]).tolist()

options_global = build_dropdown(df_global)
options_regional = build_dropdown(df_regional)

dropdown_options = options_global + options_regional

# Map displayed text -> (dataset_source, index)
lookup = {}
for i, txt in enumerate(options_global):
    lookup[txt] = ("global", i)
for i, txt in enumerate(options_regional):
    lookup[txt] = ("regional", i)

# Sidebar controls
st.sidebar.header("Controls")

selected_song = st.sidebar.selectbox(
    "Select a song",
    dropdown_options
)

dataset_source, query_index = lookup[selected_song]

# Mode availability depends on dataset
modes = ["Music / Beats", "Vibe / Mood"]
if dataset_source == "global":
    modes.append("Genre")

mode = st.sidebar.radio("Recommendation type", modes)

top_k = st.sidebar.slider("Number of recommendations", 3, 10, 5)

# Core recommendation logic
def recommend_cosine(df, matrix, query_idx, top_k):
    q = matrix[query_idx].reshape(1, -1)
    scores = cosine_similarity(q, matrix)[0]
    scores[query_idx] = -1  # exclude itself
    idx = np.argsort(scores)[::-1][:top_k]
    return df.loc[idx, [
        "track_name", "artist_name", "album_name"
    ]]

def recommend_by_genre(df, query_idx, top_k):
    genre = df.loc[query_idx, "track_genre"]
    cands = df[df["track_genre"] == genre]
    cands = cands[cands.index != query_idx]
    return cands.sort_values("popularity", ascending=False).head(top_k)[[
        "track_name", "artist_name", "album_name"
    ]]

# ROUTING (AUTOMATIC)
if dataset_source == "global":
    df_active = df_global
    music_matrix = music_global
    vibe_matrix = vibe_global
else:
    df_active = df_regional
    music_matrix = music_regional
    vibe_matrix = vibe_regional

# RUN
if st.sidebar.button("Recommend"):
    st.subheader("Recommended Songs")

    if mode == "Genre":
        results = recommend_by_genre(df_active, query_index, top_k)
    elif mode == "Music / Beats":
        results = recommend_cosine(df_active, music_matrix, query_index, top_k)
    else:
        results = recommend_cosine(df_active, vibe_matrix, query_index, top_k)

    results = results.reset_index(drop=True)
    results.index = results.index + 1
    st.dataframe(results, use_container_width=True)



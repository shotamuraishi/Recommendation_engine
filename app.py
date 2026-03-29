import streamlit as st
import numpy as np
import pandas as pd
import random
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MaxAbsScaler


st.set_page_config(layout="wide")

if 'selected_movie_indices' not in st.session_state:
    st.session_state.selected_movie_indices = []

# ----- Data -----
df_movie = pd.read_csv('data/movies.csv', index_col='imdbId')
df_movie = df_movie.reset_index(drop=True)

# Clean the data set
df_movie['year'] = df_movie['title'].str.extract(r'\((\d{4})\)')
df_movie['title'] = df_movie['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
df_movie['year'] = pd.to_numeric(df_movie['year'], errors='coerce').fillna(0).astype(int)


# Get dummy values
genre_features = df_movie['genres'].str.get_dummies(sep='|')

# Normalize year
scaler = MaxAbsScaler()
year_features = scaler.fit_transform(df_movie[['year']])

# Vectorize words
tfidf = TfidfVectorizer(stop_words='english', max_features=100)
tfidf_features = tfidf.fit_transform(df_movie['title']).toarray()

# Combine all features
features = np.hstack([genre_features.values, year_features, tfidf_features])

# Make a model
random.seed(42)
model_knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
model_knn.fit(features)

def show_recommendations(prefix=""):
    # When the list is empty
    if not st.session_state.selected_movie_indices:
        random_movies = df_movie.sample(10, random_state=42)
        
        for movie_id, movie in random_movies.iterrows():
            @st.write(f"movie_id: {movie_id}, type: {type(movie_id)}")
            with st.container():
                col1, col2= st.columns([0.8, 0.2], vertical_alignment="center")
                with col1:
                    st.markdown(f"**{movie['title']}**")
                    st.caption(f"Released year: {movie['year']}")
                    st.caption(f"Genres: {movie['genres'].replace('|', ', ')}")
                with col2:
                    if st.button("Add", key=f"{prefix}rand_add_{movie_id}", use_container_width=True):
                        st.session_state.selected_movie_indices.append(movie_id)
                        st.toast(f"Added {movie['title']}!")
                        st.rerun()
            st.divider()
        return

    # When movies are in the list
    else:
        # Calculate the 'User Profile' vector
        user_profile = features[st.session_state.selected_movie_indices].mean(axis=0).reshape(1, -1)
        
        # Find nearest neighbors
        distances, indices = model_knn.kneighbors(user_profile, n_neighbors=10)

        st.write("### ✨ Recommended for You")
        st.caption("Based on your current selections")

        count = 0
        for pos in indices.flatten():
            movie_id = int(pos)
            
            if movie_id not in st.session_state.selected_movie_indices:
                movie = df_movie.iloc[movie_id]
                
                with st.container():
                    col1, col2= st.columns([0.8, 0.2], vertical_alignment="center")
                    
                    # Movie information
                    with col1:
                        st.markdown(f"**{movie['title']}**")
                        st.caption(f"Released: {movie['year']}")
                        st.caption(f"Genres: {movie['genres'].replace('|', ', ')}")
                    
                    # Add button
                    with col2:
                        if st.button("Add", key=f"{prefix}rec_{movie_id}", use_container_width=True):
                            st.session_state.selected_movie_indices.append(movie_id)
                            st.toast(f"Added {movie['title']}!")
                            st.rerun()
                
                st.divider()
                count += 1

            # Only display 10 movies at that same time    
            if count > 10:
                break



col_main, col_sidebar = st.columns([0.6, 0.4], gap="large")
# Search engine
with col_main:
    st.header("🔍 Search Movies")
    query = st.text_input("Enter title or year", placeholder="e.g. Toy Story", key="search_input")

    if query:
        results = df_movie[
            df_movie['title'].str.contains(query, case=False, na=False) |
            df_movie['year'].astype(str).str.contains(query, na=False)
        ].head(10)

        if not results.empty:
            st.write(f"### Results for '{query}'")
            for idx, row in results.iterrows():
                with st.container():
                    c1, c2 = st.columns([0.8, 0.2], vertical_alignment="center")
                    with c1:
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"Released: {row['year']}")
                        st.caption(f"Genres: {row['genres'].replace('|', ', ')}")
                    with c2:
                        if idx in st.session_state.selected_movie_indices:
                            st.button("Selected", key=f"search_added_{idx}", disabled=True, use_container_width=True)
                        else:
                            if st.button("Add", key=f"search_add_{idx}", use_container_width=True):
                                st.session_state.selected_movie_indices.append(idx)
                                st.rerun()
                st.divider()
        else:
            st.warning("No movies found.")
    else:
        st.info("Start searching to add movies to your list!")
        show_recommendations(prefix="main_")

# List and recommendation engine
with col_sidebar:
    #st.write(st.session_state.selected_movie_indices)
    st.subheader("📋 Your Selection")
    if not st.session_state.selected_movie_indices:
        st.info("Add movies to see recommendations!")
    else:
        for idx in st.session_state.selected_movie_indices:
            m = df_movie.iloc[idx]
            c1, c2 = st.columns([0.85, 0.15], vertical_alignment="center")
            with c1:
                st.markdown(f"**{m['title']}**")
                st.caption(f"Released: {m['year']}")
                st.caption(f"Genres: {m['genres'].replace('|', ', ')}")
            if c2.button("Del", key=f"side_del_{idx}"):
                st.session_state.selected_movie_indices.remove(idx)
                st.rerun()
        
        if st.button("Clear All", use_container_width=True):
            st.session_state.selected_movie_indices = []
            st.rerun()

        st.write("---")
        if st.session_state.selected_movie_indices:
            show_recommendations(prefix="side_")


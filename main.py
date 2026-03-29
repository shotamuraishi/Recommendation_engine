import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MaxAbsScaler
from IPython.display import display

df_movie = pd.read_csv('data/movies.csv', index_col='imdbId')
df_description = pd.read_csv('data/movies_description.csv')
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
model_knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
model_knn.fit(features)


# --- Recommendation System Logic ---
selected_movie_indices = []  # List to keep track of user selections

def show_recommendations():
    """Function to display recommendations based on past selections"""
    if not selected_movie_indices:
        return
    
    # Calculate the average vector of all selected movies
    user_profile = features[selected_movie_indices].mean(axis=0).reshape(1, -1)
    
    # Find the nearest neighbors using KNN
    distances, indices = model_knn.kneighbors(user_profile, n_neighbors=10)

    recommendations_data = []
    
    for pos in indices.flatten():
        movie_id = df_movie.index[pos]
        if movie_id not in selected_movie_indices:
            movie = df_movie.loc[movie_id]
            recommendations_data.append({
                'Title': movie['title'],
                'Year': movie['year'],
                'Genres': movie['genres'].replace('|', ', ')
            })

        if len(recommendations_data) >= 10:
            break

    print("\n[ Recommendations for You ]")
    rec_df = pd.DataFrame(recommendations_data)

    display(rec_df)


# --- Main Interaction Loop ---
print("\n--- Initial Random Recommendations ---")
print(df_movie[['title', 'genres']].sample(10))


while True:
    prompt = """
--- SELECTION MENU ---
0. Search
1. Show recommendations
q. Quit
Selection: """ 
    query = input(prompt).strip()
    if query.lower() == 'q':
        break

    # Search engine
    if query == '0':
        while True:
            search_query = input("\nEnter the movie title or released year (type 'b' to go back menu): ").strip()
            if search_query.lower() == 'b':
                break # Back to menu

            else:
                results = df_movie[
                    df_movie['title'].str.contains(search_query, case=False, na=False) |
                    df_movie['year'].astype(str).str.contains(search_query, na=False)
                ]

                if not results.empty:
                    print(f"\nSearch results of {search_query}:")
                    display_df = results[['title', 'year', 'genres']].head(10)
                    if (len(results) > 10):
                        print("\nThe search results are truncated. It displays only 10 rows.")
                        print(display_df)
                    else:
                        print(display_df)

                else:
                    print(f"\nDoes not find movies")
                    continue
    
    # Recommendation engine
    elif query == '1':
        while True:
            if selected_movie_indices:
                current_titles = df_movie.loc[selected_movie_indices, ['title']]
                print("\nYour Selections:")
                print(current_titles.to_string(index_names=True))
            else:
                print("\nYour Selections: (Empty)")

            rec_query = input("""
--- SELECTION MENU ---
* Enter the movie name or year
* Type '-ID' to delete selected movies
* Type 'clear' to delete all selected movies
* Type 'b' to go back to menu
Type here: """).strip()
            if rec_query.lower() == 'b':
                break # Back to menu

            elif rec_query.startswith('-'):
                remove_id = rec_query[1:] 
                if remove_id.isdigit():
                    idx = int(remove_id)
                    if idx in selected_movie_indices:
                        selected_movie_indices.remove(idx)
                        print(f"\nRemoved: {df_movie.loc[idx, 'title']}")
                        show_recommendations() 
                        continue
                    else:
                        print(f"\n{idx} is not in your list.")
                        continue

            elif rec_query.lower() == 'clear':
                confirm = input("\nAre you sure you want to clear all selections? (y/n): ").strip().lower()
                if confirm == 'y':
                    selected_movie_indices = [] 
                    print("\nAll selections have been cleared.")
                continue
            
            results = df_movie[df_movie['title'].str.contains(rec_query, case=False, na=False)]
            
            if results.empty:
                print("\nNo matches found.")
                continue
            
            print(f"\n--- Search Results for '{rec_query}' ---")
            print(results[['title', 'genres']].head(10))
            
            choice = input("\nEnter the left most number to select (or Enter to skip): ").strip()
            
            if choice.isdigit():
                idx = int(choice)
                if idx in results.index:
                    if idx not in selected_movie_indices:
                        selected_movie_indices.append(idx)
                        print(f"Added: {df_movie.loc[idx, 'title']}")
                        show_recommendations()
                        continue
                    else:
                        print("\nAlready selected.")
                else:
                    print(f"\nIndex {idx} is not in the search results above.")
    else:
        print("Invalid input, pls try it again")

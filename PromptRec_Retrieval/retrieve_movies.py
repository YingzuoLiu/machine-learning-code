
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import config
import numpy as np

def load_movies(path):
    # u.item file format: movie_id|movie_title|...
    df = pd.read_csv(path, sep='|', header=None, encoding='latin-1', usecols=[0,1], names=['id', 'title'])
    return df

def build_index(movies):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(movies["title"])
    return vectorizer, tfidf_matrix

def retrieve_top_k(input_text, vectorizer, tfidf_matrix, movies, k):
    input_vec = vectorizer.transform([input_text])
    scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:k]
    return movies.iloc[top_indices][["id", "title"]], scores[top_indices]

if __name__ == "__main__":
    movies = load_movies(config["movie_data_path"])
    vectorizer, tfidf_matrix = build_index(movies)

    query = "space, science fiction, mind-bending, Nolan"
    top_movies, scores = retrieve_top_k(query, vectorizer, tfidf_matrix, movies, config["top_k"])

    print("Query:", query)
    print("Top Recommendations:")
    for i, row in top_movies.iterrows():
        print(f"{row['title']}")

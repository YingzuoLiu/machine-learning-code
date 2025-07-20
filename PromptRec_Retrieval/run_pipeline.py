
from generate_keywords import extract_keywords
from retrieve_movies import load_movies, build_index, retrieve_top_k
from config import config

def full_recommendation(prompt):
    interest = extract_keywords(prompt)
    print("\n[Step 1] Extracted Interest:", interest)

    movies = load_movies(config["movie_data_path"])
    vectorizer, tfidf_matrix = build_index(movies)
    top_movies, _ = retrieve_top_k(interest, vectorizer, tfidf_matrix, movies, config["top_k"])

    print("\n[Step 2] Final Movie Recommendations:")
    for i, row in top_movies.iterrows():
        print(f"{row['title']}")

if __name__ == "__main__":
    prompt = "I loved The Matrix and Inception. Recommend something mind-blowing."
    full_recommendation(prompt)

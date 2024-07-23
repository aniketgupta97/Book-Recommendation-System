import pickle
import pandas as pd

# Load the dataset
# Ensure this matches the dataset used for training
df = pd.read_csv('books_and_genres.csv', nrows=10000)  # Adjust as needed based on memory capacity

# Load the pre-trained models from pickle files
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

with open('genre_binarizer.pkl', 'rb') as file:
    mlb = pickle.load(file)

with open('cosine_similarity.pkl', 'rb') as file:
    cosine_sim = pickle.load(file)

# Define the recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the book that matches the title
    idx = df[df['title'].str.lower() == title.lower()].index[0]

    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return df['title'].iloc[book_indices]

# Example usage of the recommendation function
recommendations = get_recommendations('the house on the borderland')
print(recommendations)



import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
df = pd.read_csv('books_and_genres.csv', nrows=10000)  # Adjust as needed based on memory capacity

# Cleaning function to remove irrelevant text parts
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove Project Gutenberg header/footer if present
    text = re.sub(r"Produced by .*?\n", "", text)
    text = re.sub(r"End of the Project Gutenberg.*", "", text, flags=re.S)

    # Remove non-alphabetic characters and multiple spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()

# Apply the cleaning function to the 'text' column
df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x) if isinstance(x, str) else "")

# Standardize genres
def clean_genres(genres):
    # Convert string representation of set to actual set
    genres = eval(genres)
    # Standardize to lowercase and strip whitespace
    return set(g.lower().strip() for g in genres)

df['genres'] = df['genres'].apply(lambda x: clean_genres(x) if isinstance(x, str) else set())

# Vectorize the cleaned text using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)  # Adjust the max_features based on your dataset size
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

# Encode genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_matrix = mlb.fit_transform(df['genres'])

# Combine the TF-IDF features and genre features
combined_matrix = hstack([tfidf_matrix, genres_matrix])

# Compute cosine similarity between books
cosine_sim = cosine_similarity(combined_matrix, combined_matrix)


#Visualiations

   # Genre Distribution Plot

plt.figure(figsize=(10, 6))
genre_counts = df['genres'].explode().value_counts().nlargest(15)  # Adjust number as needed
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis')
plt.title('Top 10 Genre Distribution in the Dataset')
plt.xlabel('Number of Books')
plt.ylabel('Genres')
plt.show()

# Recommendation function
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

# Save the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf, file)

# Save the MultiLabelBinarizer
with open('genre_binarizer.pkl', 'wb') as file:
    pickle.dump(mlb, file)

# Save the cosine similarity matrix
with open('cosine_similarity.pkl', 'wb') as file:
    pickle.dump(cosine_sim, file)


# Heatmap of Cosine Similarity (Subset to avoid memory issues)
plt.figure(figsize=(12, 10))
subset_cosine_sim = cosine_sim[:100, :100]  # Limit to first 100 rows and columns for example
sns.heatmap(subset_cosine_sim, cmap='coolwarm', square=True, xticklabels=df['title'][:100], yticklabels=df['title'][:100])
plt.title('Subset of Cosine Similarity Heatmap (First 100 Books)')
plt.show()


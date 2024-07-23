import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

# Function to clean text data
def clean_text(text):
    text = re.sub(r"Produced by .*?\n", "", text)
    text = re.sub(r"End of the Project Gutenberg.*", "", text, flags=re.S)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# Function to clean and standardize genres
def clean_genres(genres):
    genres = eval(genres)
    return set(g.lower().strip() for g in genres)

# Load and preprocess dataset
df = pd.read_csv('books_and_genres.csv')
df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x) if isinstance(x, str) else "")
df['genres'] = df['genres'].apply(lambda x: clean_genres(x) if isinstance(x, str) else set())

# Encode genres
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

# Define the pipeline with TfidfVectorizer and a classifier (e.g., MultinomialNB) using best parameters
best_params = {'clf__estimator__alpha': 0.1, 'tfidf__max_features': 1000, 'tfidf__ngram_range': (1, 1)}

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=best_params['tfidf__max_features'], ngram_range=best_params['tfidf__ngram_range'])),
    ('clf', MultiOutputClassifier(MultinomialNB(alpha=best_params['clf__estimator__alpha'])))
])

# Train the final model on the entire dataset
pipeline.fit(df['cleaned_text'], y)

# Save the final model
with open('final_model.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

# Save the MultiLabelBinarizer
with open('mlb.pkl', 'wb') as file:
    pickle.dump(mlb, file)

# Predict genres for all books in the dataset
df['predicted_genres'] = list(pipeline.predict(df['cleaned_text']))

# Save the dataframe with predictions for recommendation purposes
df.to_csv('books_with_predictions.csv', index=False)

print("Model training complete, and predictions saved to 'books_with_predictions.csv'.")

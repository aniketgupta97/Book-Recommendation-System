import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle
import re

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
df = pd.read_csv('books_and_genres.csv', nrows=1000)  # Further reducing the number of rows
df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x) if isinstance(x, str) else "")
df['genres'] = df['genres'].apply(lambda x: clean_genres(x) if isinstance(x, str) else set())

# Use a smaller subset of the data for parameter tuning
df_subset = df.sample(n=500, random_state=42)  # Reducing the subset size

# Encode genres
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df_subset['genres'])

# Define the pipeline with TfidfVectorizer and a classifier (e.g., MultinomialNB)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultiOutputClassifier(MultinomialNB()))
])

# Define a simplified parameter grid for tuning
parameters = {
    'tfidf__max_features': [1000, 2000],  # Further reducing the parameter space
    'tfidf__ngram_range': [(1, 1)],
    'clf__estimator__alpha': [0.1, 1.0]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=1, verbose=1)  # Limiting to 1 job to save memory
grid_search.fit(df_subset['cleaned_text'], y)

# Print best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Save the best model
best_model = grid_search.best_estimator_
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

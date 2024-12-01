#FastText#
import pandas as pd
import numpy as np
import preprocess
from gensim.models.fasttext import FastText
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def preprocess_and_vectorize(dataset_path):
    df_new = pd.read_csv(dataset_path)
    df_new['label'] = df_new['label'].map({'real': 1, 'fake': 0})
    df_new.dropna(subset=['label'], inplace=True)
    df_new['processed_tweet'] = df_new['tweet'].apply(preprocess.simple_preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(df_new['processed_tweet'], df_new['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train.tolist(), X_test.tolist(), y_train, y_test, vectorizer  # Return lists of processed text


# Assuming preprocess_and_vectorize now returns raw text data for X_train and X_test
X_train_raw, X_test_raw, y_train, y_test, vectorizer = preprocess_and_vectorize('dataset.csv')

# Tokenize the tweets for FastText
X_train_tokenized = [tweet.split() for tweet in X_train_raw]
X_test_tokenized = [tweet.split() for tweet in X_test_raw]


# Train a FastText model
model_ft = FastText(vector_size=100, window=5, min_count=1, sentences=X_train_tokenized, epochs=10)

# Function to vectorize tweets by averaging word vectors
def vectorize_tweets(model, tweets):
    vectorized = []
    for tweet in tweets:
        vectorized.append(
            np.mean([model.wv[word] for word in tweet if word in model.wv] or [np.zeros(model.vector_size)], axis=0)
        )
    return np.array(vectorized)

# Vectorize the training and testing data
X_train_vectorized = vectorize_tweets(model_ft, X_train_tokenized)
X_test_vectorized = vectorize_tweets(model_ft, X_test_tokenized)

# Since FastText doesn't directly provide a way to classify, use another classifier on top of the embeddings
from sklearn.linear_model import LogisticRegression

# Train a classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vectorized, y_train)

# Predict on the test set
predictions = clf.predict(X_test_vectorized)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

#accuracy, precision, recall, f1, conf_matrix

print("FastText Confusion Matrix:", conf_matrix)
print("FastText Accuracy:", accuracy)
print("FastText Precision:", precision)
print("FastText Recall:", recall)
print("FastText F1 Score:", f1)

model_ft.save('fasttext_model.model')

# Placeholder for best model and metrics
best_model = None
best_accuracy = 0
best_params = {}

# Define parameter grid
vector_sizes = [50, 100, 150]
window_sizes = [3, 5, 7]
epochs = [5, 10, 15]

for vector_size in vector_sizes:
    for window in window_sizes:
        for epoch in epochs:
            # Train FastText model
            model_ft = FastText(vector_size=vector_size, window=window, min_count=1, sentences=X_train_tokenized, epochs=epoch)

            # Vectorize the training and testing data, similar to the previous step
            X_train_vectorized = vectorize_tweets(model_ft, X_train_tokenized)
            X_test_vectorized = vectorize_tweets(model_ft, X_test_tokenized)

            # Train a classifier
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train_vectorized, y_train)

            # Predict on the test set and evaluate
            predictions = clf.predict(X_test_vectorized)
            accuracy = accuracy_score(y_test, predictions)
            
            # Update best model if it has the highest accuracy so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_ft
                best_params = {'vector_size': vector_size, 'window': window, 'epochs': epoch}

print(f"Best Parameters: {best_params}")
print(f"Best Accuracy: {best_accuracy}")


# Preprocessing and TF-IDF Vectorization Code
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
# import nltk
# #nltk.download('punkt')
# from nltk.corpus import stopwords

def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  
                           u"\U0001F300-\U0001F5FF"  
                           u"\U0001F680-\U0001F6FF"  
                           u"\U0001F1E0-\U0001F1FF"  
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def simple_preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = remove_emojis(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_and_vectorize(dataset_path):
    df_new = pd.read_csv(dataset_path)
    df_new['label'] = df_new['label'].map({'real': 1, 'fake': 0})
    df_new.dropna(subset=['label'], inplace=True)
    df_new['processed_tweet'] = df_new['tweet'].apply(simple_preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(df_new['processed_tweet'], df_new['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    # Save the TF-IDF vectorizer and trained models
    dump(vectorizer, 'tfidf_vectorizer.joblib')
    return X_train_tfidf, X_test_tfidf, y_train, y_test

# Call the function with your dataset path
X_train_tfidf, X_test_tfidf, y_train, y_test = preprocess_and_vectorize('dataset.csv')



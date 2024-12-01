import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load Dataset
dataset_path = 'dataset2.csv'  # Update this path
data = pd.read_csv(dataset_path)

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '<url>', text)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '<emoji>', text)
    text = re.sub(r'#\S+', '<hashtag>', text)
    text = re.sub(r'\@\S+', '<mention>', text)
    return text

data['tweet'] = data['tweet'].apply(preprocess_text)

# Split Dataset into Training, Validation, and Testing
X = data['tweet']
y = data['label']
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# TF-IDF Vectorization for DNN
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_val_tfidf = tfidf_vectorizer.transform(X_val).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

# Dump the fitted TF-IDF Vectorizer for later use
dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

# Encode Labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

def return_encoders():
    return y_train_encoded, y_val_encoded, y_test_encoded

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_tfidf, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

def return_tensors():
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor   

# DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

def return_dataset():
    return train_dataset, val_dataset, test_dataset

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def return_dataloaders():
    return train_loader, val_loader, test_loader



#####FastText###########

from gensim.models import FastText

# Combine training and validation sets for FastText training
combined_texts = pd.concat([X_train, X_val])

# Tokenize texts
tokenized_texts = [text.split() for text in combined_texts]

# Train FastText model
fasttext_model = FastText(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Function to create matrix representation for a given sentence
def sentence_to_matrix(sentence, fasttext_model, max_len, vector_size):
    matrix = np.zeros((max_len, vector_size))
    tokens = sentence.split()
    for i, token in enumerate(tokens[:max_len]):
        if token in fasttext_model.wv:
            matrix[i] = fasttext_model.wv[token]
    return matrix

# Determine max length of sentences in the dataset
max_inp_len = max(len(text.split()) for text in combined_texts)

# Create matrix representations for each sentence in the datasets
X_train_matrices = np.array([sentence_to_matrix(sentence, fasttext_model, max_inp_len, 100) for sentence in X_train])
X_val_matrices = np.array([sentence_to_matrix(sentence, fasttext_model, max_inp_len, 100) for sentence in X_val])
X_test_matrices = np.array([sentence_to_matrix(sentence, fasttext_model, max_inp_len, 100) for sentence in X_test])


def return_matrices():
    return X_train_matrices, X_val_matrices, X_test_matrices
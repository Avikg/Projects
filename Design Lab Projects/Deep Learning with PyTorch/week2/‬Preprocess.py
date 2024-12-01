import pandas as pd
from sklearn.model_selection import train_test_split
import re

# Load the dataset
dataset_path = 'Dataset3.csv'
df = pd.read_csv(dataset_path)

# Task 1: Split the dataset into training, validation, and test sets (80/10/10)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

# Preprocessing function for social media posts (Task 2)
def preprocess_text(text):
    # Handling URLs
    text = re.sub(r'http\S+|www\.\S+', '<URL>', text)
    # Handling emojis - this is a placeholder; you might need a library like 'emoji' to handle emojis effectively
    text = re.sub(r'(:\S+:)', '<EMOJI>', text)
    # Handling hashtags
    text = re.sub(r'#(\S+)', r'HASHTAG_\1', text)
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '<url>', text)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '<emoji>', text)
    text = re.sub(r'#\S+', '<hashtag>', text)
    text = re.sub(r'\@\S+', '<mention>', text)
    return text
    return text

# Apply preprocessing to each dataset split
train_df['tweet'] = train_df['tweet'].apply(preprocess_text)
val_df['tweet'] = val_df['tweet'].apply(preprocess_text)
test_df['tweet'] = test_df['tweet'].apply(preprocess_text)

# Save the split and preprocessed datasets to CSV files
train_df.to_csv('train_dataset.csv', index=False)
val_df.to_csv('validation_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

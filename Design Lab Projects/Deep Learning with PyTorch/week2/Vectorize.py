import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel

def get_model_and_tokenizer(model_name):
    if model_name in ['bert-base-uncased', 'bert-base-cased']:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

def encode_texts(texts, tokenizer, model):
    encoded_batch = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    input_ids = encoded_batch['input_ids']
    attention_mask = encoded_batch['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def process_and_save_embeddings(file_path, output_file_prefix, model_name):
    model, tokenizer = get_model_and_tokenizer(model_name)
    df = pd.read_csv(file_path)
    embeddings_list = []
    texts = df['tweet'].tolist()
    labels = df['label'].values
    
    batch_size = 10
    for start_index in range(0, len(df), batch_size):
        end_index = start_index + batch_size
        texts = df['tweet'][start_index:end_index].tolist()
        embeddings = encode_texts(texts, tokenizer, model)
        
        embeddings_np = embeddings.cpu().numpy()
        embeddings_list.append(embeddings_np)
    
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    # Replace slashes in the model name with underscores for the filename
    filename_safe_model_name = model_name.replace("/", "_")
    np.save(f'{output_file_prefix}_{filename_safe_model_name}_embeddings.npy', all_embeddings)
    # Save the labels alongside embeddings
    np.save(f'{output_file_prefix}_labels.npy', labels)

    
# Adjust these paths according to your file locations
train_dataset_path = 'train_dataset.csv'
validation_dataset_path = 'validation_dataset.csv'
test_dataset_path = 'test_dataset.csv'

# Corrected function calls for process_and_save_embeddings
process_and_save_embeddings(train_dataset_path, 'train', 'bert-base-cased')
process_and_save_embeddings(validation_dataset_path, 'validation', 'bert-base-cased')
process_and_save_embeddings(test_dataset_path, 'test', 'bert-base-cased')

process_and_save_embeddings(train_dataset_path, 'train', 'bert-base-uncased')
process_and_save_embeddings(validation_dataset_path, 'validation', 'bert-base-uncased')
process_and_save_embeddings(test_dataset_path, 'test', 'bert-base-uncased')

process_and_save_embeddings(train_dataset_path, 'train', 'digitalepidemiologylab/covid-twitter-bert')
process_and_save_embeddings(validation_dataset_path, 'validation', 'digitalepidemiologylab/covid-twitter-bert')
process_and_save_embeddings(test_dataset_path, 'test', 'digitalepidemiologylab/covid-twitter-bert')

process_and_save_embeddings(train_dataset_path, 'train', 'Twitter/twhin-bert-base')
process_and_save_embeddings(validation_dataset_path, 'validation', 'Twitter/twhin-bert-base')
process_and_save_embeddings(test_dataset_path, 'test', 'Twitter/twhin-bert-base')

process_and_save_embeddings(train_dataset_path, 'train', 'sarkerlab/SocBERT-base')
process_and_save_embeddings(validation_dataset_path, 'validation', 'sarkerlab/SocBERT-base')
process_and_save_embeddings(test_dataset_path, 'test', 'sarkerlab/SocBERT-base')


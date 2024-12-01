import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNN, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)  
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x
    


def load_data(embeddings_file, labels_file):
    X = np.load(embeddings_file, allow_pickle=True)
    y = np.load(labels_file, allow_pickle=True)
    
    # Ensure X is properly formatted as a float32 array
    if isinstance(X[0], np.ndarray):
        X = np.vstack(X).astype(np.float32)
    else:
        # If X is a list of arrays with varying lengths, additional handling may be needed
        X = np.array([np.array(xi) for xi in X], dtype=object).astype(np.float32)
    
    # Encode labels as integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded.astype(np.int64)

def create_objective(embeddings_file, labels_file):
    def objective(trial):
        # Define hyperparameters
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        
        # Load data (assuming embeddings and labels are prepared as .npy files)
        #embeddings_file = 'train_bert-base-cased_embeddings.npy'  # Adjust path as necessary
        #labels_file = 'train_labels.npy'          # Adjust path as necessary
        X, y = load_data(embeddings_file, labels_file)      # Your load_data function here
        
        # Data preparation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_data = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
        
        # Model setup
        model = DNN(X_train.shape[1], hidden_dim, len(set(y)))  # Your DNN class here
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(10):  # Example: 10 epochs
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Validation
        model.eval()
        all_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
        
        accuracy = accuracy_score(y_val, all_preds)
        return -accuracy  # Minimize the negative accuracy
    return objective

def run_optimization_for_embeddings(embeddings_name):
    embeddings_file = f'train_{embeddings_name}_embeddings.npy'  # Adjust path as necessary
    labels_file = 'train_labels.npy'                             # Adjust path as necessary
    study_name = f'study_{embeddings_name}'
    
    study = optuna.create_study(study_name=study_name)
    study.optimize(create_objective(embeddings_file, labels_file), n_trials=20)

    print(f"Results for {embeddings_name}:")
    print("  Best accuracy: ", -study.best_trial.value)
    print("  Best parameters: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("\n")

# List of embeddings names corresponding to the models
embeddings_names = [
    'bert-base-cased',
    'bert-base-uncased',
    'digitalepidemiologylab_covid-twitter-bert',
    'sarkerlab_SocBERT-base',
    'Twitter_twhin-bert-base'
]

for embeddings_name in embeddings_names:
    run_optimization_for_embeddings(embeddings_name)

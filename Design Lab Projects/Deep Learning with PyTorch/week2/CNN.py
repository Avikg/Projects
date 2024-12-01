import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

def load_data(embeddings_file, labels_file, fixed_seq_length=512, embedding_dim=768):
    """
    Load embeddings and labels, pad embeddings to uniform sequence length, and encode labels.
    
    Args:
        embeddings_file (str): File path for the embeddings numpy array.
        labels_file (str): File path for the labels numpy array.
        fixed_seq_length (int, optional): Desired fixed sequence length for all embeddings.
        embedding_dim (int, optional): Dimensionality of the embeddings.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the padded embeddings tensor and the encoded labels tensor.
    """
    embeddings = np.load(embeddings_file, allow_pickle=True)
    labels = np.load(labels_file, allow_pickle=True)

    # Ensure each embedding has the correct shape and pad sequences
    padded_embeddings = np.zeros((len(embeddings), fixed_seq_length, embedding_dim), dtype=np.float32)
    for i, embedding in enumerate(embeddings):
        embedding = np.array(embedding)
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=-1)
        length = min(fixed_seq_length, embedding.shape[0])
        padded_embeddings[i, :length, :embedding.shape[1]] = embedding[:length, :]

    # Encode labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    embeddings_tensor = torch.tensor(padded_embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)

    return embeddings_tensor, labels_tensor




class CNN1D(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_size, output_dim):
        super(CNN1D, self).__init__()
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=filter_size, padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(num_filters, output_dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reorder dimensions to [batch, embedding_dim, seq_len]
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(2)  # Remove the last dimension
        x = self.fc(x)
        return x

def create_objective(embeddings_file, labels_file):
    def objective(trial):
        # Hyperparameters
        num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])
        filter_size = trial.suggest_int("filter_size", 2, 5)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

        # Data preparation
        X, y = load_data(embeddings_file, labels_file)
        embedding_dim = X.shape[2]
        output_dim = len(torch.unique(y))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        # Model setup
        model = CNN1D(embedding_dim, num_filters, filter_size, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(10):  # Adjust epochs based on dataset size and convergence
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.tolist())
        
        accuracy = accuracy_score(y_test.tolist(), predictions)
        return accuracy

    return objective





# Main execution part
embeddings_names = [
    'bert-base-cased',
    'bert-base-uncased',
    'digitalepidemiologylab_covid-twitter-bert',
    'sarkerlab_SocBERT-base',
    'Twitter_twhin-bert-base'
]

for embeddings_name in embeddings_names:
    embeddings_file = f'train_{embeddings_name}_embeddings.npy'
    labels_file = 'train_labels.npy'
    study = optuna.create_study(direction="maximize")
    study.optimize(create_objective(embeddings_file, labels_file), n_trials=20)
    print(f"Results for {embeddings_name} with 1D-CNN:")
    print("  Best accuracy: ", study.best_value)
    print("  Best parameters: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("\n")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder


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

def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred

def print_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    confusion = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}\nConfusion Matrix:\n{confusion}")

def train_and_evaluate_with_best_parameters(embeddings_name, num_filters, filter_size, lr):
    embeddings_file = f'train_{embeddings_name}_embeddings.npy'
    labels_file = 'train_labels.npy'
    X, y = load_data(embeddings_file, labels_file, fixed_seq_length=512, embedding_dim=768)
    output_dim = len(set(y.numpy()))

    # Data preparation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    # Model setup
    model = CNN1D(768, num_filters, filter_size, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(10):  # Adjust epochs based on dataset size and convergence
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    y_true, y_pred = evaluate_model(model, test_loader)
    print_metrics(y_true, y_pred)

    # Save the model
    torch.save(model.state_dict(), f"{embeddings_name}_cnn_best_model.pth")

# Define best parameters obtained from the optimization
best_parameters = {
    'bert-base-cased': {'num_filters': 128, 'filter_size': 5, 'lr': 0.0021297846295038813},
    'bert-base-uncased': {'num_filters': 128, 'filter_size': 5, 'lr': 0.0028577873657319192},
    'digitalepidemiologylab_covid-twitter-bert': {'num_filters': 32, 'filter_size': 5, 'lr': 0.006282271397599633},
    'sarkerlab_SocBERT-base': {'num_filters': 128, 'filter_size': 5, 'lr': 0.0015299744123931903},
    'Twitter_twhin-bert-base': {'num_filters': 64, 'filter_size': 4, 'lr': 0.005905767768356304}
}

# Train and evaluate each model with its best parameters
for embeddings_name, params in best_parameters.items():
    print(f"\nResults for {embeddings_name} with 1D-CNN:")
    train_and_evaluate_with_best_parameters(embeddings_name, **params)

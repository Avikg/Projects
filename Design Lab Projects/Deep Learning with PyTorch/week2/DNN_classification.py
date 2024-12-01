import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder

# DNN model definition remains the same
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

# Load data function remains the same
def load_data(embeddings_file, labels_file):
    X = np.load(embeddings_file, allow_pickle=True)
    y = np.load(labels_file, allow_pickle=True)
    X = np.vstack(X).astype(np.float32)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded

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

def print_metrics(accuracy, precision, recall, f1, confusion):
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Confusion Matrix:\n{confusion}")

# Updated function to use specific best parameters
def train_and_evaluate(embeddings_name, hidden_dim, lr):
    embeddings_file = f'train_{embeddings_name}_embeddings.npy'
    labels_file = 'train_labels.npy'

    X, y = load_data(embeddings_file, labels_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    output_dim = len(set(y))
    model = DNN(X_train.shape[1], hidden_dim, output_dim)
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

    # Evaluation
    y_true, y_pred = evaluate_model(model, test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    confusion = confusion_matrix(y_true, y_pred)

    print_metrics(accuracy, precision, recall, f1, confusion)

    # Save the model
    torch.save(model.state_dict(), f"{embeddings_name}_best_model.pth")

# Define the best hyperparameters for each embeddings type
best_params = {
    'bert-base-cased': {'hidden_dim': 256, 'lr': 0.00262411133527765},
    'bert-base-uncased': {'hidden_dim': 256, 'lr': 0.002916251679018978},
    'digitalepidemiologylab_covid-twitter-bert': {'hidden_dim': 256, 'lr': 0.0017802124705326963},
    'sarkerlab_SocBERT-base': {'hidden_dim': 256, 'lr': 0.0009731524811667679},
    'Twitter_twhin-bert-base': {'hidden_dim': 256, 'lr': 0.0011934451361869656}
}

# Train and evaluate each model with its best parameters
for embeddings_name, params in best_params.items():
    print(f"Processing: {embeddings_name}")
    train_and_evaluate(embeddings_name, **params)
    print("\n")

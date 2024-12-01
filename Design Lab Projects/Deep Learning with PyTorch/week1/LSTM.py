import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import PreprocessAndVectorize

# Convert matrices and labels into PyTorch tensors
X_train_matrices, X_val_matrices, X_test_matrices=PreprocessAndVectorize.return_matrices()
y_train_encoded, y_val_encoded, y_test_encoded=PreprocessAndVectorize.return_encoders()
X_train_tensor = torch.tensor(X_train_matrices, dtype=torch.float)
X_val_tensor = torch.tensor(X_val_matrices, dtype=torch.float)
X_test_tensor = torch.tensor(X_test_matrices, dtype=torch.float)

y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

# Prepare TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

def returnDataset():
    return train_dataset, val_dataset

batch_size = 64  # Adjustable based on computational resources
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

def returnLoader():
    return train_loader, val_loader

class TextLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(TextLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 because of bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_length, embedding_dim]
        _, (hidden, _) = self.lstm(x)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

# Model Initialization
embedding_dim = 100  # FastText embedding dimension
hidden_dim = 256  # Hidden layer dimension
output_dim = 2  # Number of output classes
num_layers = 2  # Number of LSTM layers
dropout = 0.5  # Dropout rate

model = TextLSTM(embedding_dim, hidden_dim, output_dim, num_layers, dropout).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5  # Number of epochs
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer.zero_grad()
        output = model(texts)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}')


import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import PreprocessAndVectorize

# Convert matrices and labels into PyTorch tensors
X_train_matrices, X_val_matrices, X_test_matrices=PreprocessAndVectorize.return_matrices()
y_train_encoded, y_val_encoded, y_test_encoded=PreprocessAndVectorize.return_encoders()
X_train_tensor = torch.tensor(X_train_matrices, dtype=torch.float)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_matrices, dtype=torch.float)
y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

# Prepare TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

def returnDataset():
    return train_dataset, val_dataset

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

class TextCNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes, filter_sizes, num_filters, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Model initialization
embedding_dim = 100  # Should match the FastText model's dimension
num_embeddings = X_train_tensor.size(1) + 1  # Plus one for padding index
num_classes = 2  # Assuming binary classification
filter_sizes = [2, 3, 4]  # Different kernel sizes
num_filters = 100  # Number of filters per kernel size

def returnEmbedings():
    return embedding_dim, num_embeddings, num_classes, filter_sizes, num_filters

model = TextCNN(num_embeddings, embedding_dim, num_classes, filter_sizes, num_filters).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 5
for epoch in range(epochs):
    model.train()
    for texts, labels in train_loader:
        texts, labels = texts.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer.zero_grad()
        output = model(texts)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

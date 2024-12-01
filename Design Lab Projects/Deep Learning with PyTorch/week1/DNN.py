import torch
import torch.nn as nn
import torch.optim as optim
import PreprocessAndVectorize

class DNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout_rate=0.5):
        super(DNNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layers[1], output_size)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x
    
X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = PreprocessAndVectorize.return_tensors()
train_dataset, val_dataset, test_dataset=PreprocessAndVectorize.return_dataset()
train_loader, val_loader, test_loader=PreprocessAndVectorize.return_dataloaders()
    

input_size = X_train_tensor.shape[1]
output_size = 2  # Adjust based on your task, e.g., 2 for binary classification
hidden_layers = [512, 256]  # Example: two hidden layers with 512 and 256 units

model = DNNModel(input_size, output_size, hidden_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')

model.eval()
total_correct = 0
total_count = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)
        total_count += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_count
print(f'Validation Accuracy: {accuracy:.4f}')

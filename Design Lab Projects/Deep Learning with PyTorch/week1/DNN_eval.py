import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import PreprocessAndVectorize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = PreprocessAndVectorize.return_tensors()
train_dataset, val_dataset, test_dataset=PreprocessAndVectorize.return_dataset()
train_loader, val_loader, test_loader=PreprocessAndVectorize.return_dataloaders()

class DynamicDNN(nn.Module):
    def __init__(self, input_dim, layer_sizes, dropout_rate):
        super(DynamicDNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, layer_sizes[i]))
            else:
                self.layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        self.out = nn.Linear(layer_sizes[-1], 2)  # Assuming binary classification
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Dropout):
                if i != 0:  # Apply activation and dropout after linear layers except the first iteration
                    x = layer(x)
        x = self.out(x)
        return x


def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 4)
    layer_sizes = [trial.suggest_int(f'n_units_l{i}', 32, 512) for i in range(n_layers)]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])

    input_dim = X_train_tensor.shape[1]  # This should be 5000 based on TF-IDF vectorization
    model = DynamicDNN(input_dim, layer_sizes, dropout_rate).to(device)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(10):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validation loop
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)  # Adjust based on your computational resources

best_trial = study.best_trial
best_model = DynamicDNN(
    input_dim=X_train_tensor.shape[1],
    layer_sizes=[best_trial.params[f'n_units_l{i}'] for i in range(best_trial.params['n_layers'])],
    dropout_rate=best_trial.params['dropout_rate']
).to(device)

# Save the best model
torch.save(best_model.state_dict(), 'best_dnn_model.pth')

print("Best model saved as 'best_dnn_model.pth'")

# Outputting the best trial's parameters and accuracy for confirmation
print(f"Best trial accuracy: {study.best_trial.value}")
print("Best trial parameters:")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")

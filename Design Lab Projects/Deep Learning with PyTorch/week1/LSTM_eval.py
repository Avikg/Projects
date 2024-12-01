import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.metrics import accuracy_score
import LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset, val_dataset=LSTM.returnDataset()
train_loader, val_loader=LSTM.returnLoader()

# Define the LSTM Model
class TextLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(TextLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)
    
def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])  # Reduced batch size
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    
    model = TextLSTM(embedding_dim=100, hidden_dim=hidden_dim, output_dim=2, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Adjust number of epochs if necessary
    for epoch in range(10):
        model.train()
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
      # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # Adjust the number of trials based on computational resources

# After optimization, reinitialize the best model to save it
best_params = study.best_trial.params
best_model = TextLSTM(
    embedding_dim=100, 
    hidden_dim=best_params['hidden_dim'], 
    output_dim=2, 
    num_layers=best_params['num_layers'], 
    dropout=best_params['dropout']
).to(device)

# Assuming the best model is retrained here or you decide to save without retraining
torch.save(best_model.state_dict(), 'best_lstm_model.pth')
print("Best LSTM model saved as 'best_lstm_model.pth'")


# Best trial results
print("Best trial:")
print(f"Accuracy: {study.best_trial.value}")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")
best_params = study.best_trial.params
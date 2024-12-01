import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.metrics import accuracy_score
import PreprocessAndVectorize
import CNN
from CNN import TextCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim, num_embeddings, num_classes, filter_sizes, num_filters=CNN.returnEmbedings()
train_dataset, val_dataset=CNN.returnDataset()
epochs = 5
criterion = nn.CrossEntropyLoss()

def objective(trial):
    # Ensure the device is defined within the objective if not globally defined
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_filters = trial.suggest_categorical("num_filters", [50, 100, 150])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    
    # Adjust filter_sizes to be suggested as strings and parse them back into tuples
    filter_sizes_str = trial.suggest_categorical("filter_sizes", ["(2, 3, 4)", "(3, 4, 5)"])
    filter_sizes = eval(filter_sizes_str)  # Parse the string back into a tuple
    
    # Model specific hyperparameters
    padding = trial.suggest_categorical("padding", ["valid", "same"])

    # Create the model
    model = TextCNN(num_embeddings, embedding_dim, num_classes, filter_sizes, num_filters, dropout=dropout_rate).to(device)

    # Define optimizer
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # Train the model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(texts)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            output = model(texts)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # Adjust n_trials based on computational resources

print("Best trial:")
print(study.best_trial.params)
best_trial = study.best_trial

# Recreate the best model using the best trial parameters
filter_sizes = eval(best_trial.params['filter_sizes'])  # Ensure to safely parse string to tuple if stored as string
model = TextCNN(
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
    num_classes=num_classes,
    filter_sizes=filter_sizes,
    num_filters=best_trial.params['num_filters'],
    dropout=best_trial.params['dropout_rate']
).to(device)

# Save the best model to disk
torch.save(model.state_dict(), 'best_cnn_model.pth')

print("Best CNN model saved as 'best_cnn_model.pth'")
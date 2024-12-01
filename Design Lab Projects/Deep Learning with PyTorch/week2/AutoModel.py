import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, ClassLabel, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate


# Function to check command-line arguments
def check_arguments():
    if len(sys.argv) != 3:
        print("Usage: python3 AutoModel.py <bert_model_name> <dataset_path>")
        sys.exit(1)
    return sys.argv[1], sys.argv[2]

bert_model_name, dataset_path = check_arguments()

# Load the dataset
data = pd.read_csv(dataset_path)

# Ensure labels are integers
label_mapping = {'fake': 0, 'real': 1}  # Adjust as necessary
data['label'] = data['label'].map(label_mapping).astype(int)

# Split the dataset
train_val, test = train_test_split(data, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1111, random_state=42)

# Convert to Hugging Face datasets
features = Dataset.from_pandas(train).features
features['label'] = ClassLabel(names=['fake', 'real'])
train_dataset = Dataset.from_pandas(train, features=features)
val_dataset = Dataset.from_pandas(val, features=features)
test_dataset = Dataset.from_pandas(test, features=features)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

# Initialize the data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# Tokenization and encoding the dataset
def preprocess_function(examples):
    return tokenizer(examples["tweet"], truncation=True, padding=True, max_length=512)

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)

# Define accuracy metric
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,  # Ensure tokenizer is passed for correct collation
    data_collator=data_collator,  # Use the data collator
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained("./best_model")
tokenizer.save_pretrained("./best_model")

# Evaluation
print("Evaluation on test set:")
results = trainer.evaluate(tokenized_test_dataset)

# Print evaluation results
for key, value in results.items():
    print(f"{key}: {value}")

# Make predictions for the classification report and confusion matrix
predictions = trainer.predict(tokenized_test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Classification Report
report = classification_report(true_labels, preds, target_names=['Fake', 'Real'])

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, preds)

# Accuracy
accuracy = accuracy_score(true_labels, preds)

print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy: {:.2f}%".format(accuracy * 100))

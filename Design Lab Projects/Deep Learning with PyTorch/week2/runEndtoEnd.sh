#!/bin/bash

# Set these variables to the correct paths for your setup
DATASET_PATH="dataset3.csv"
BERT_MODEL_NAME="bert-base-uncased"  # Example BERT model
OUTPUT_DIR="output/"  # Directory to store any output files

# Assuming the preprocess script does not take command-line arguments
echo "Running Preprocessing..."
python3 Preprocess.py

echo "Running Vectorization..."
python3 Vectorize.py

echo "Training DNN model..."
python3 DNN.py

echo "Training CNN model..."
python3 CNN.py

echo "Evaluating DNN model..."
python3 DNN_Classification.py

echo "Evaluating CNN model..."
python3 CNN_Classification.py 

echo "Running AutoModel..."
python3 AutoModel.py "$BERT_MODEL_NAME" "$DATASET_PATH"

echo "All tasks completed."

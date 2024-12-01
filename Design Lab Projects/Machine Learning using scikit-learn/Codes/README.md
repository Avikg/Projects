# Social Media Post Classification for Fake News Detection

## Dataset Overview
This project uses a dataset from the Constraint@AAAI-2021 shared task on COVID19-related fake news detection in English. It comprises 10,600 samples sourced from various social media platforms, with 5,545 labeled as real and 5,055 as fake.

## Prerequisites
- Python 3.8+
- Libraries: scikit-learn, numpy, pandas, scipy, seaborn, matplotlib

## Installation
Set up your environment and install the required libraries:
```
python3 -m venv env
source env/bin/activate  # Use `env\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Data Preparation Steps

### Task 1: Dataset Split
- Utilize scikit-learn's `train_test_split` with the shuffle option.
- Split the dataset into 80% training, 10% validation, and 10% test sets.
- Export these splits into three separate CSV files.

### Task 2: Preprocessing Social Media Posts
- Retain crucial non-textual information such as emojis and hashtags to preserve the post's context and sentiment.
- Preprocessing includes:
  - Removing unnecessary characters and noise.
  - Handling or preserving emojis and hashtags based on their relevance to the content's sentiment.

### Task 3: Vector Representation
- Convert text data into vectors using TF-IDF (Term Frequency-Inverse Document Frequency) representation.
- This process transforms the input sentences into a format suitable for machine learning model input.

## Python Scripts
- `preprocess.py`: Splits the dataset as per Task 1.
- `preprocess.py`: Conducts preprocessing steps outlined in Task 2.
- `preprocess.py`: Transforms preprocessed text into TF-IDF vectors (Task 3).
- `<particular_model_name>.py`: Contains code for training six different ML models specified in the assignment.
- Each model (KNN, Logistic Regression, SVM, K-Means, Neural Networks, FastText) has a dedicated script for hyperparameter tuning and training.

## Running the Code
First run preprocess.py and then the corresponding model_name.py for ex. LogisticRegression.py

## Report and Models
The final submission includes a detailed report covering hyperparameter details and performance metrics (accuracy, F1-score, precision, recall) obtained from `Report.pdf`. The report also provides a Drive link containing the TF-IDF Vectorizer model and weights for all ML models in a scikit-learn compatible format.
# Machine Learning Assignment (Week 1) - CS69011 Computing Lab II

## Overview

This repository contains the solution to **Week 1 Machine Learning Assignment** as part of the **CS69011 Computing Lab II** course. The assignment aims to classify social media posts into **real** or **fake** using various machine learning models and preprocessing techniques.

---

## Directory Structure

```
.
├── dataset.csv                # Input dataset for the assignment
├── preprocess.py              # Script for social media text preprocessing
├── LogisticRegression.py      # Script for Logistic Regression model
├── KNN.py                     # Script for K-Nearest Neighbors model
├── SVM.py                     # Script for Support Vector Machine model
├── KMeans.py                  # Script for K-Means Clustering
├── NeuralNetwork.py           # Script for Neural Network model
├── FastText.py                # Script for FastText model
├── Report.pdf                 # Detailed report with evaluation metrics and hyperparameters
├── README.md                  # This file
```

---

## Tasks

### Dataset: `dataset.csv`
- The dataset contains 10,600 samples, with:
  - 5,545 labeled as real.
  - 5,055 labeled as fake.

### Preprocessing: `preprocess.py`
- Cleans and preprocesses social media posts to retain critical information like emojis, URLs, and hashtags.
- Outputs a processed dataset ready for vectorization or feeding into models.

### Machine Learning Models
1. **Logistic Regression**: Implemented in `LogisticRegression.py`.
2. **K-Nearest Neighbor (KNN)**: Implemented in `KNN.py`.
3. **Support Vector Machine (SVM)**: Implemented in `SVM.py`.
4. **K-Means Clustering**: Implemented in `KMeans.py`.
5. **Neural Network**: Implemented in `NeuralNetwork.py`.
6. **FastText**: Implemented in `FastText.py`. Takes raw text as input instead of TF-IDF vectors.

---

## How to Run

### Preprocessing
Run the preprocessing script to clean the dataset:
```bash
python preprocess.py
```

### Training Models
Run the respective Python script to train a specific model:
- Logistic Regression:
  ```bash
  python LogisticRegression.py
  ```
- KNN:
  ```bash
  python KNN.py
  ```
- SVM:
  ```bash
  python SVM.py
  ```
- K-Means:
  ```bash
  python KMeans.py
  ```
- Neural Network:
  ```bash
  python NeuralNetwork.py
  ```
- FastText:
  ```bash
  python FastText.py
  ```

### Output
- Each script will generate:
  - Model performance metrics (accuracy, precision, recall, F1-score).
  - Confusion matrix.
  - Best hyperparameters used for the model.

---

## Report
- `Report.pdf` includes:
  - Preprocessing techniques.
  - Hyperparameter tuning steps.
  - Evaluation metrics for each model.

---

## Libraries Used
1. **Python Libraries**:
   - scikit-learn
   - numpy
   - pandas
   - scipy
   - seaborn
   - matplotlib
2. **FastText**:
   - Follow the [FastText Supervised Tutorial](https://fasttext.cc/docs/en/supervised-tutorial.html).

---

## Submission
- Ensure the following are included:
  - `dataset.csv`
  - All Python scripts.
  - `Report.pdf`

---

## Author

**Avik Pramanick**  
**Roll No:** 23CS60R78  
**Course:** CS69011 Computing Lab II  

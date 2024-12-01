# Deep Learning Assignments - Week 1 & Week 2 (CS69011 Computing Lab II)

## Overview

This repository contains solutions for the **Week 1** and **Week 2** Deep Learning Assignments as part of the **CS69011 Computing Lab II** course. The assignments aim to classify social media posts into **real** or **fake** categories using various deep learning models, progressing from basic architectures to advanced BERT-based models.

---

## Repository Structure

### Week 1 Directory

```
Week1/
├── dataset2.csv                     # Input dataset for preprocessing and training
├── PreprocessAndVectorize.py        # Script for preprocessing and vectorization
├── DNN.py                           # Script for training Deep Neural Networks
├── CNN.py                           # Script for training Convolutional Neural Networks
├── LSTM.py                          # Script for training Long Short-Term Memory Networks
├── DNN_eval.py                      # Evaluation script for Deep Neural Networks
├── CNN_eval.py                      # Evaluation script for Convolutional Neural Networks
├── LSTM_eval.py                     # Evaluation script for Long Short-Term Memory Networks
├── RunEval.py                       # General evaluation script
├── runEndtoEnd.sh                   # Shell script for end-to-end execution
├── Models/                          # Directory for storing model checkpoints
├── Report.pdf                       # Detailed report for Week 1 assignment
```

### Week 2 Directory

```
Week2/
├── Dataset3.csv                     # Input dataset for preprocessing and training
├── train_dataset.csv                # Training split
├── validation_dataset.csv           # Validation split
├── test_dataset.csv                 # Test split
├── Preprocess.py                    # Script for preprocessing social media posts
├── Vectorize.py                     # Script for BERT-based vectorization
├── DNN.py                           # Script for training Deep Neural Networks
├── CNN.py                           # Script for training Convolutional Neural Networks
├── AutoModel.py                     # Script for training BERT-based models
├── DNN_classification.py            # Evaluation script for DNN
├── CNN_classification.py            # Evaluation script for CNN
├── RunEval.py                       # General evaluation script
├── runEndtoEnd.sh                   # Shell script for end-to-end execution
├── Models/                          # Directory for storing model checkpoints
├── Report.pdf                       # Detailed report for Week 2 assignment
```

---

## Dataset

- **Source**: Constraint@AAAI-2021 shared task on COVID-19 fake news detection.
- **Size**: 10,600 samples, with:
  - 5,545 labeled as real.
  - 5,055 labeled as fake.
- **Splits**:
  - **Train**: 80%
  - **Validation**: 10%
  - **Test**: 10%

---

## Tasks and Models

### Week 1 Tasks
1. **Preprocessing and Vectorization**: `PreprocessAndVectorize.py`
2. **Training Models**:
   - **DNN**: `DNN.py`
   - **CNN**: `CNN.py`
   - **LSTM**: `LSTM.py`
3. **Evaluation**:
   - **DNN**: `DNN_eval.py`
   - **CNN**: `CNN_eval.py`
   - **LSTM**: `LSTM_eval.py`
4. **End-to-End Execution**: `runEndtoEnd.sh`

### Week 2 Tasks
1. **Preprocessing**: `Preprocess.py`
2. **Vectorization**: `Vectorize.py` (BERT-based models)
3. **Training Models**:
   - **DNN**: `DNN.py`
   - **CNN**: `CNN.py`
   - **BERT-based Models**: `AutoModel.py`
4. **Evaluation**:
   - **DNN**: `DNN_classification.py`
   - **CNN**: `CNN_classification.py`
   - General Evaluation: `RunEval.py`
5. **End-to-End Execution**: `runEndtoEnd.sh`

---

## How to Run

### Preprocessing
- Week 1:
  ```bash
  python Week1/PreprocessAndVectorize.py
  ```
- Week 2:
  ```bash
  python Week2/Preprocess.py
  ```

### Vectorization (Week 2)
Run the vectorization script for BERT-based models:
```bash
python Week2/Vectorize.py --bert_model <bert-model-name>
```

### Training
- Week 1:
  ```bash
  python Week1/DNN.py
  python Week1/CNN.py
  python Week1/LSTM.py
  ```
- Week 2:
  ```bash
  python Week2/DNN.py
  python Week2/CNN.py
  python Week2/AutoModel.py
  ```

### Evaluation
Evaluate models using respective scripts:
- Week 1:
  ```bash
  python Week1/DNN_eval.py
  python Week1/CNN_eval.py
  python Week1/LSTM_eval.py
  ```
- Week 2:
  ```bash
  python Week2/DNN_classification.py
  python Week2/CNN_classification.py
  python Week2/RunEval.py --model <model-name> --test_path <test-dataset-path>
  ```

### End-to-End Execution
Run all tasks sequentially:
- Week 1:
  ```bash
  bash Week1/runEndtoEnd.sh
  ```
- Week 2:
  ```bash
  bash Week2/runEndtoEnd.sh
  ```

---

## Reports

1. **Week 1**: `Week1/Report.pdf`
2. **Week 2**: `Week2/Report.pdf`

Each report contains:
- Preprocessing and vectorization details.
- Hyperparameter tuning strategies.
- Evaluation metrics (accuracy, F1-score, precision, recall).
- Observations and conclusions.

---

## Libraries Used

1. **Core Libraries**:
   - PyTorch
   - scikit-learn
   - numpy
   - pandas
   - scipy
2. **FastText**:
   - Used for embedding generation in Week 1.
3. **HuggingFace Transformers**:
   - Used for BERT-based models in Week 2.

---

## Author

**Avik Pramanick**  
**Roll No:** 23CS60R78  
**Course:** CS69011 Computing Lab II  

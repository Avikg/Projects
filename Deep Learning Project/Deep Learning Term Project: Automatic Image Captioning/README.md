# Deep Learning Term Project: Automatic Image Captioning

## Overview

This project involves building encoder-decoder models for **Automatic Image Captioning**. The goal is to generate meaningful captions for images using deep learning architectures such as CNN-RNN and Vision Transformer (ViT)-based models. The project is divided into two parts, with separate implementations for each model.

---

## Directory Structure

```
.
├── team_id_26_a.ipynb         # Jupyter Notebook for Part A (CNN-RNN implementation)
├── team_id_26_b.ipynb         # Jupyter Notebook for Part B (Vision Transformer implementation)
├── team_id_26_report.pdf      # Project report with methodology, results, and analysis
├── ReadMe.txt                 # Instructions and project overview (this file)
```

---

## Objectives

1. **Part A**: Implement a **CNN-based encoder** and an **RNN-based decoder** to generate captions for images.
2. **Part B**: Implement a **Vision Transformer (ViT)-based encoder** with a custom decoder for caption generation.
3. Evaluate the models using standard metrics:
   - **CIDEr**
   - **ROUGE-L**
   - **SPICE**

---

## Files Description

1. **team_id_26_a.ipynb**:
   - Implements the CNN-RNN model for image captioning.
   - Includes preprocessing, model creation, training, and evaluation.

2. **team_id_26_b.ipynb**:
   - Implements the Vision Transformer (ViT) encoder with a decoder.
   - Includes preprocessing, model creation, training, and evaluation.
   - Uses a pretrained ViT model for encoding images.

3. **team_id_26_report.pdf**:
   - Detailed project report.
   - Includes methodology, architecture diagrams, results, evaluation metrics, and analysis.

4. **ReadMe.txt**:
   - This file containing instructions and project details.

---

## Evaluation Metrics

The models were evaluated on the following metrics:
- **CIDEr**: Captures consensus between generated and reference captions.
- **ROUGE-L**: Measures overlap between generated and reference captions.
- **SPICE**: Evaluates semantic content and structure.

---

## Instructions to Run

1. Download the dataset from the provided link:
   [Dataset Download](https://drive.google.com/file/d/1FMVcFM78XZE1KE1rIkGBpCdcdI58S1LB/view?usp=sharing)

2. Open the respective notebooks in Jupyter or Colab.

3. Follow the step-by-step instructions in:
   - **team_id_26_a.ipynb** for the CNN-RNN model.
   - **team_id_26_b.ipynb** for the ViT-based model.

4. Ensure the required Python libraries are installed:
   - PyTorch
   - Transformers (for Part B)
   - scikit-learn
   - NumPy
   - Matplotlib

5. Execute each cell in the notebooks sequentially to preprocess the data, train the models, and evaluate results.

---

## Deliverables

- **team_id_26_a.ipynb**: Part A (CNN-RNN model)
- **team_id_26_b.ipynb**: Part B (ViT-based model)
- **team_id_26_report.pdf**: Project report detailing the methodologies and results.

---

## Authors

- **Team Members**:
  - **Avik Pramanick**
  - **Dipan Mondal**
  - **Rajanyo Paul**

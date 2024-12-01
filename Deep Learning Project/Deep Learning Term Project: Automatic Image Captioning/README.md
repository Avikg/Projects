# Deep Learning Term Project: Automatic Image Captioning

## Overview

This project focuses on building **encoder-decoder models** for **Automatic Image Captioning** using Deep Learning. The task is to generate meaningful captions for input images, leveraging various model architectures, including CNN-RNN and Vision Transformer-based encoders with appropriate decoders.

---

## Objectives

1. **Part A**: Implement a **CNN-based encoder** and an **RNN-based decoder** to generate captions for images.
2. **Part B**: Implement a **Vision Transformer (ViT)-based encoder** and a custom decoder for generating captions.
3. Evaluate the models using standard metrics:
   - **CIDEr**
   - **ROUGE-L**
   - **SPICE**

---

## Dataset

### Source
The dataset can be accessed from the following link:  
[Download Dataset](https://drive.google.com/file/d/1FMVcFM78XZE1KE1rIkGBpCdcdI58S1LB/view?usp=sharing)

### Format
- Images and corresponding captions are provided.
- Preprocessing includes resizing images and tokenizing captions.

---

## Subtasks

### Part A: CNN-RNN Encoder-Decoder
1. **Encoder**:
   - Build a **Convolutional Neural Network (CNN)** encoder for feature extraction from images.
2. **Decoder**:
   - Implement an **RNN-based decoder** to generate captions from image embeddings.
3. **Output**: Generated captions for the test set.

### Part B: Vision Transformer Encoder-Decoder
1. **Encoder**:
   - Use a **Vision Transformer (ViT)** as the image encoder.
   - Suggested model: `vit-small-patch16-224` from [HuggingFace](https://huggingface.co/WinKawaks/vit-small-patch16-224).
2. **Decoder**:
   - Choose and implement a suitable text decoder for caption generation.
3. **Constraints**:
   - Ensure the model uses less than 15GB of GPU memory (e.g., on Google Colab’s T4 GPU).
4. **Output**: Generated captions for the test set.

---

## Evaluation Metrics

The models will be evaluated based on the following metrics:
- **CIDEr**: Capturing consensus between generated and reference captions.
- **ROUGE-L**: Measuring overlap between generated and reference captions.
- **SPICE**: Evaluating semantic content and structure.

---

## Deliverables

1. **Python Notebooks**:
   - `team_id_<num>_a.ipynb`: Notebook for Part A.
   - `team_id_<num>_b.ipynb`: Notebook for Part B.
2. **README**:
   - Instructions for running the notebooks.
   - Additional project details (included in this file).
3. **Project Report**:
   - Max 4 pages.
   - Sections:
     - **Methodology**: Detailed explanation and diagrams for each model.
     - **Results**: Evaluation metrics for the test set and analysis.
   - Named `team_id_<num>_report.pdf`.

---

## Instructions to Run

### Part A
1. Preprocess the dataset:
   ```bash
   python PreprocessAndVectorize.py --data_path dataset/
   ```
2. Train the CNN-RNN model:
   ```bash
   python CNN_RNN_Model.py
   ```
3. Evaluate the model:
   ```bash
   python Evaluate.py --model_path checkpoints/CNN_RNN.pth --test_data test/
   ```

### Part B
1. Preprocess the dataset:
   ```bash
   python Preprocess.py --data_path dataset/
   ```
2. Train the Vision Transformer-based model:
   ```bash
   python ViT_Transformer_Model.py
   ```
3. Evaluate the model:
   ```bash
   python Evaluate.py --model_path checkpoints/ViT_Transformer.pth --test_data test/
   ```

### Generated Captions
Both notebooks will generate captions for the test set in the last cell.

---

## General Guidelines

1. Pre-trained encoders and decoders can be used for **Part B**, but they must be fine-tuned on the given dataset.
2. Do not use end-to-end pre-trained models available on platforms like HuggingFace.
3. Follow these phases in your code:
   - Preprocessing
   - Model Creation
   - Training (with validation after each epoch)
   - Evaluation on the test set
4. Use either **PyTorch** or **TensorFlow** for implementation.
5. The final outputs must be included in the submitted notebooks.

---

## References

- **Related Papers**:
  - [A PyTorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
  - [Vision Transformer Documentation](https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder)
  - [RNN Tutorials on Kaggle](https://www.kaggle.com/code/kanncaa1/recurrent-neural-network-with-pytorch)
- **Additional Resources**:
  - [Image Captioning YouTube Playlist](https://www.youtube.com/watch?v=y2BaTt1fxJU&list=PLCJHEFznK8ZybO3cpfWf4gKbyS5VZgppW&index=1)
  - [ViT Model on HuggingFace](https://huggingface.co/WinKawaks/vit-small-patch16-224)

---

## Authors

- **Team Members**:
  - **Avik Pramanick**
  - **Dipan Mondal**
  - **Rajanyo Paul**

# MTP Project: Classification of TSS Signals for Association with hg-19 CAGE Tags

## Overview

This project focuses on identifying transcription start site (TSS) signals associated with CAGE tags in the human genome (hg-19). The study leverages advanced machine learning techniques, including Bi-LSTMs, to predict whether TSS signals are associated with capping sites, which have implications in RNA stability and disease mechanisms.

---

## Objectives

1. **Identify TSS Signals**: Locate motifs like TATA boxes, Initiators, Promoter Upstream Elements (UPE), and Promoter Downstream Elements (DPE) in the genome.
2. **Classify Capping Sites**: Predict TSS signals as positive (associated with CAGE tags) or negative (not associated).
3. **Enhance Understanding of RNA Capping**: Provide insights into the mechanisms of RNA stability and its role in cellular processes and disease pathways.
4. **Develop Machine Learning Pipelines**: Use alignment-free embeddings and Bi-LSTMs for sequence classification.

---

## Dataset

### Sources
1. **Genome Data**: `hg19.fa` (FASTA file)
2. **CAGE Tags**: `hg19.cage_peak_phase1and2combined_coord.bed`

### Processed CSV Columns
- **Nucleotide sequence**: Extracted 500 bp upstream and downstream of TSS signals.
- **Chromosome (chrom)**: Chromosome ID.
- **Motif**: Type of TSS signal (e.g., TATA+UPE, INR+DPE).
- **Label**: `1` (positive) or `0` (negative) based on association with CAGE tags.
- **Motif start**: Start position of the motif.
- **Motif end**: End position of the motif.

---

## Methodology

### 1. TSS Signal Identification
- Identify TSS motifs using the following combinations:
  - **Initiator + Promoter Downstream Element (DPE)**
  - **TATA Box + Promoter Upstream Element (UPE)**

### 2. Neighborhood Analysis
- Detect CAGE tags within 500 bp upstream or downstream of the TSS signals.

### 3. Embedding Generation
- Generate embeddings using alignment-free techniques:
  - Convert 500 bp sequences into 49 tokens using a sliding window (size 20, step 10).
  - Map each token into a 32-dimensional embedding.

### 4. Bi-LSTM Classification
- Train a Bi-LSTM model with the embeddings to classify sequences as positive or negative based on their association with CAGE tags.

---

## Web Server for TSS Analysis

The **CAGE Tag classification system** is deployed at:  
**[CAGE Tag Web Server](https://cosmos.iitkgp.ac.in/CAGETag/)**  

### Implementation Details
The web server was implemented using:
- **Backend**:
  - PHP: Handles requests, runs the Bi-LSTM classification model, and communicates with the server.
- **Frontend**:
  - **HTML5**: For structuring the webpage.
  - **CSS**: For styling and enhancing user interface design.
  - **JavaScript**: For client-side interactivity and AJAX requests.
- **Database**: Stores processed TSS signal data and query logs.
- **Deployment Environment**:
  - Hosted on the **COSMOS Lab Server** at IIT Kharagpur.
  - Model predictions are served using PHP scripts interfacing with Python for ML inference.

---

## Results

The Bi-LSTM model achieved the following:
- **Accuracy**: Detailed metrics to be included based on training logs.
- **Applicability**: Predicted TSS signals for experimental validation in vivo.

---

## Future Work

1. Validate predicted positive TSS signals experimentally in vivo.
2. Extend the methodology to genomes beyond hg-19.
3. Develop models for predicting genome uncapping and recapping sites.
4. Integrate RNA expression data for improved prediction accuracy.

---

# Additional Project: DeepPROTECTNeo

### Objective
DeepPROTECTNeo is a web server developed for neoepitope prediction with applications in reverse vaccinology and personalized cancer vaccine design.

### Features
1. **Context-Aware Analysis**: Transformer with cross-attention for TCR-epitope binding.
2. **High Accuracy**: AUC 0.6634, AUPRC 0.6759, validated on TESLA benchmark.
3. **Pipeline**:
   - Input: NGS BAM files, HLA typing data, TCR CDR3 sequences, tumor peptides.
   - Output: Neoepitope candidates, binding scores, and vaccine suggestions.
4. **Impact**: Enhances neoepitope discovery, aiding personalized cancer vaccine design.

### Web Server for DeepPROTECTNeo

The **DeepPROTECTNeo prediction system** is deployed at:  
**[DeepPROTECTNeo Web Server](https://cosmos.iitkgp.ac.in/DeepPROTECTNeo/)**  

### Implementation Details
The web server was implemented using:
- **Backend**:
  - PHP: Processes user-uploaded NGS files and interfaces with the prediction model.
- **Frontend**:
  - **HTML5, CSS, JavaScript**: For user interaction, data entry, and visualization of results.
- **Additional Tools**:
  - **NetMHCpan**: For predicting tumor-derived peptides.
  - **TRUST4**: For identifying TCR CDR3 β sequences.

---

## Journal Submission and Review Status

Both projects, **CAGE Tag Analysis** and **DeepPROTECTNeo**, are under paper submission to reputed journals and currently **under review**. The methodologies and results presented in these projects have been prepared for peer-reviewed publication.

---

## Acknowledgements

I would like to express my gratitude to:
- **Supervisor**: Dr. Pralay Mitra, Associate Professor, IIT Kharagpur.
- **Mentor**: Dibya Kanti Halder, Research Scholar.
- COSMOS Lab and CCDS Lab members for their support.
- IIT Kharagpur for facilities, resources, and fellowship support.

## For Codes

Codes are password protected as there is secrecy of paper publication. For more details contact the undersigned.

**Avik Pramanick**  
**Roll No:** 23CS60R78  
**Email:** avik.pramanick@gmail.com

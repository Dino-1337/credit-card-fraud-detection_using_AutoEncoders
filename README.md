# Credit Card Fraud Detection using Autoencoders

## Project Overview
This project focuses on detecting fraudulent credit card transactions using a deep learning approach.  
The dataset is highly imbalanced, with fraud cases being less than 0.2% of all transactions.  
The main objective is to build a model that can detect frauds with high recall while keeping false positives low.

## Why Autoencoders?
Autoencoders are effective for fraud detection because they learn to reconstruct normal (non-fraud) transactions accurately.  
When fraudulent transactions are passed through the model, the reconstruction error is significantly higher, making them easier to flag as anomalies without requiring large amounts of labeled fraud data.

## Methods Used
- Data preprocessing and feature scaling  
- Handling class imbalance using SMOTE (Synthetic Minority Oversampling Technique)  
- Building a 7-layer denoising autoencoder for anomaly detection  
- Threshold tuning for classification  
- Model evaluation on a separate test set

## Evaluation Metrics
- Accuracy: **98.30%**  
- Recall : **88.78%**  
- Precision : **8%**  
- F1-score : **0.15** 
## Dataset
The dataset used is the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).  
It contains 284,807 transactions with anonymized features (V1–V28), along with `Time`, `Amount`, and `Class` labels.

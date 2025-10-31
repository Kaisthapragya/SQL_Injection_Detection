# SQL_Injection_Detection
Hybrid SQL-injection detector: character n-gram logistic regression + character-level CNN, averaged and thresholded from validation.

This project detects Structured Query Language (SQL) injection attempts in text inputs using a hybrid approach: 

(1) A classical model that learns character-level n-gram patterns via term-frequency–inverse-document-frequency (TF-IDF) features fed to logistic regression.
(2) A lightweight character-level convolutional neural network that learns token-free signal directly from raw characters. 

Their probabilities are combined by a simple average ensemble. A decision threshold is chosen on a validation split to balance precision and recall, and the code also saves clear evaluation plots (confusion matrices, receiver-operating-characteristic curves, and precision–recall curves).

Dataset: [https://www.kaggle.com/datasets/sajid576/sql-injection-dataset]

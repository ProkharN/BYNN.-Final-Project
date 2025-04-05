# BYNN.-Final-Project

Project: Sentiment Analysis with Custom Neural Networks

Description:
This project implements sentiment analysis on movie reviews using neural network architectures (LSTM, CNN, and FeedForward) built in PyTorch. The project was developed as part of the "Build Your Own Neural Network" course at University Potsdam. The goal is to classify IMDB reviews as positive or negative, and the project includes several experiments that explore different hyperparameter settings to analyze their impact on model performance.

Folder Structure:
- main.py: The main project script that contains the implementation, including data loading, model definitions, training, and evaluation routines.
- training_results: This folder contains CSV documents that store the results of all experiments. These CSV files summarize metrics such as test loss, accuracy, precision, recall, and F1 score for each model configuration.
- data: This folder holds the dataset used in the project, including the IMDB reviews for both training and testing. The dataset is organized into subdirectories (e.g., "train/pos", "train/neg", "test/pos", "test/neg") for easy access during data loading.

Usage:
1. Ensure the dataset is properly stored in the "data" folder following the required structure.
2. Run the main.py script to start the training process and generate the experiment results.
3. Review the CSV files in the "training_results" folder to analyze model performance under different experimental settings.

Amazon Fine Food Review Sentiment Analysis
1. Project Overview
This project developed a highly accurate sentiment analysis tool to classify customer reviews for Amazon Fine Food products. Using an advanced deep learning model, the tool can reliably predict whether a review is positive or negative, providing valuable insights into customer satisfaction.

2. The Goal
The primary objective was to build a machine learning model that automates the process of understanding customer sentiment from unstructured text data. The final output is a user-friendly tool that a business can use to instantly analyze customer feedback.

3. Methodology
4. 
The solution uses a Recurrent Neural Network (RNN), specifically an LSTM (Long Short-Term Memory) model. This deep learning approach is highly effective for sequential data like text. The key steps were:

Data Preprocessing: Tokenizing and padding text reviews to a uniform length.

Model Training: Training the LSTM model on a large dataset of customer reviews and their corresponding sentiment scores.

Model Evaluation: Testing the model on unseen data to ensure its accuracy and reliability.

4. Model Performance
The final model achieved an impressive accuracy of 94.3% on a test set of over 110,000 reviews. This means the model correctly predicted the sentiment of more than 9 out of every 10 reviews it analyzed.

The confusion matrix below provides a more detailed look at the model's performance:

Predicted Negative	Predicted Positive
Actual Negative	20,901	4,011
Actual Positive	2,454	86,315
Correct Predictions: The model correctly identified 20,901 negative and 86,315 positive reviews.

Incorrect Predictions: Only a small number of reviews were misclassified, confirming the model's high reliability.

5. How to Use the Tool
This project includes a command-line tool that allows you to instantly get sentiment predictions.

Prerequisites:

Python 3.7+

The required libraries listed in requirements.txt

Instructions:

Save the predict_cli.py, rnn_model.h5, and tokenizer.pkl files to the same folder.

Open your terminal or command prompt.

Navigate to the folder and run the following command:

Bash

python predict_cli.py
Type a review when prompted to get the sentiment prediction.

6. Repository Contents
SentimentAnalysis.ipynb: The Jupyter Notebook containing the full project code.

Reviews.csv: The original dataset used for training.

rnn_model.h5: The trained deep learning model.

tokenizer.pkl: The saved tokenizer object.

history.json: The training history of the model.

predict_cli.py: The command-line tool for sentiment prediction.

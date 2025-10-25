# Twitter Sentiment Analysis (2022)

Analyze public sentiment in tweets using Natural Language Processing and Machine Learning. This project processes a large dataset and trains a model to classify tweet sentiments as positive or negative.

## 📋 Project Overview

- Uses a large tweet dataset to classify sentiment as either positive or negative.
- Applies text preprocessing and feature engineering for effective model training.
- Implements machine learning (Logistic Regression) for sentiment classification.
- Generates accuracy metrics for both training and testing data.

## 📂 Data

- Dataset: `training.1600000.processed.noemoticon.csv` (Kaggle)
  - Columns: target, ids, date, flag, user, text
  - `target`: sentiment label (0 = negative, 1 = positive)
- Download from Kaggle; see notebook instructions for usage.

## 🛠️ Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, NLTK

Install dependencies:

## 📝 How It Works

1. **Data Loading**: Loads CSV file into a DataFrame.
2. **Text Preprocessing**: Cleans tweets (removes noise, stopwords, applies stemming).
3. **Feature Generation**: Uses `TfidfVectorizer` for text features.
4. **Model Training**: Splits data (train-test) and fits a Logistic Regression classifier.
5. **Evaluation**: Reports accuracy for prediction on test and training data.

## 🚀 Quickstart

Clone and run the notebook using Jupyter/Colab.


Open `Main.ipynb` and run all cells.

## 🧪 Results

- Training accuracy: ~79.8%
- Testing accuracy: ~78.0%
- Classifier: Logistic Regression (with TF-IDF features).

## 🗂 Project Structure


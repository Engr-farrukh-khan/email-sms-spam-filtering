# Email/SMS Spam Filtering Using Machine Learning

This project is a machine-learning solution for detecting spam in both email and SMS messages. It uses Natural Language Processing (NLP) techniques to classify messages as either spam or non-spam (ham). The project is deployed as a web app using **Streamlit**, allowing users to interact with the model and classify messages in real-time.

## Features

- **Real-time Spam Classification**: Input any email/SMS text and instantly know if it's spam or non-spam.
- **User-Friendly UI**: An interactive and easy-to-use interface built with Streamlit.
- **Scalability**: The model is designed to handle large datasets and can be integrated into real-world email/SMS filtering systems.
- **End-to-End Pipeline**: Complete workflow from data preprocessing, model training, to web deployment.

## Tech Stack

- **Python**: Core language for data processing, model building, and deployment.
- **Jupyter Notebook**: Used for initial data exploration, model training, and evaluation.
- **Streamlit**: To deploy the app and create an interactive web interface.
- **pandas**: Data manipulation and preprocessing.
- **scikit-learn**: Machine learning library for model training and evaluation.
- **nltk**: Natural Language Processing library used for tokenization, stopword removal, and text processing.

## Project Workflow

### 1. Data Collection
The dataset contains labeled email and SMS messages, indicating whether each message is spam or not. You can download datasets like the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) for training the model.

### 2. Data Preprocessing
- **Text Cleaning**: Messages are preprocessed to remove irrelevant characters, numbers, and punctuation.
- **Tokenization**: The text is split into individual words or tokens.
- **Stopword Removal**: Common words that do not contribute to spam detection (e.g., "the", "is", "at") are removed.
- **Vectorization**: The text is transformed into numerical features using methods like TF-IDF (Term Frequency-Inverse Document Frequency).

### 3. Model Training
- The dataset is split into training and testing sets.
- A classification algorithm such as **Logistic Regression** is trained using the TF-IDF vectorized features.
- Performance is evaluated using metrics like accuracy, precision, recall, and F1 score.

### 4. Streamlit Web App Deployment
The model is deployed using **Streamlit**, allowing users to input email/SMS messages and get predictions in real-time. The app provides a simple and interactive interface for testing the spam filter.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/email-sms-spam-filtering.git


## Results
**Accuracy**: The model accurately classified messages as spam or non-spam on the test data.
**Precision & Recall**: The model demonstrated strong performance in correctly classifying spam messages while minimizing false positives.
**Metric	Value**
  Accuracy	98.5%
  Precision	96.7%
  Recall	97.3%
  F1 Score	97.0%

## Results

- **Accuracy**: The model accurately classified messages as spam or non-spam on the test data.
- **Precision & Recall**: The model demonstrated strong performance in correctly classifying spam messages while minimizing false positives.

| Metric      | Value       |
| ----------- | ----------- |
| Accuracy    | 98.5%       |
| Precision   | 96.7%       |
| Recall      | 97.3%       |
| F1 Score    | 97.0%       |


## Future Improvements

- **Model Optimization**: Experimenting with different machine learning algorithms, such as Support Vector Machines (SVM) or neural networks, to further improve accuracy.
- **Integration**: Incorporating the model into existing email or SMS filtering systems to provide real-time spam filtering at scale.
- **Advanced NLP**: Implementing more advanced NLP techniques, such as word embeddings or deep learning models like LSTMs or transformers, for more sophisticated spam detection.

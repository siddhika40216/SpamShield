# SpamShield - SMS & Email Spam Detection

## Overview

SpamShield is a machine learning based web application that detects whether a given SMS or email message is spam or legitimate.

The application uses Natural Language Processing (NLP) techniques and machine learning algorithms to analyze text patterns and classify messages. It performs text preprocessing, TF-IDF vectorization, and uses a trained classification model to provide real-time spam predictions with confidence probability.

The model is deployed as an interactive web application using Streamlit.

## Technology Used

### Programming Language
- Python

### Machine Learning & NLP
- Scikit-learn
- NLTK
- TF-IDF Vectorizer
- Logistic Regression
- Naive Bayes

### Data Processing
- Pandas
- NumPy

### Deployment
- Streamlit

### Database
- SQLite

## Features

- SMS spam detection
- Email spam detection
- Text preprocessing using NLP
- TF-IDF based feature extraction
- Machine learning based classification
- Spam probability prediction
- Streamlit web interface
- SQLite based prediction storage

### Data Collection

The project uses SMS and email spam datasets containing messages labeled as spam and legitimate.

The datasets were combined and preprocessed to build a unified spam detection model capable of classifying both SMS and email text.

### Data Cleaning and Preprocessing
The data was cleaned by handling null and duplicate values. The data was then preprocessed by converting the text into tokens, removing special characters, stop words and punctuation, and stemming the data. The data was also converted to lowercase before preprocessing.

### Exploratory Data Analysis
Exploratory Data Analysis was performed to gain insights into the dataset. The count of characters, words, and sentences was calculated for each message. The correlation between variables was also calculated, and visualizations were created using pyplots, bar charts, pie charts, 5 number summaries, and heatmaps. Word clouds were also created for spam and non-spam messages, and the most frequent words in spam texts were visualized.

### Model Building and Selection
Multiple classifier models were tried, including NaiveBayes, random forest, KNN, decision tree, logistic regression, ExtraTreesClassifier, and SVC. The best classifier was chosen based on precision, with a precision of 100% achieved.

## Model Building and Selection

Different machine learning algorithms including Multinomial Naive Bayes, Bernoulli Naive Bayes, and Logistic Regression were evaluated.

The final model was selected based on performance metrics such as accuracy, precision, and recall.

Voting Classifier was used as the final classifier and trained on TF-IDF transformed text features.

## Web Deployment

The trained model was deployed using Streamlit.

The application provides:
- Text input for SMS or email messages
- Spam/Not Spam prediction
- Spam probability score
- Clean interactive user interface

Prediction records are stored using SQLite database for future reference.

## Demo

Live Demo:
https://spamshield-by-siddhika.streamlit.app/

## Usage
To use the SMS Spam Detection model on your own machine, follow these steps:

+ Clone this repository.
+ Install the required Python packages using 
```
pip install -r requirements.txt.
```
+ Run the model using 
```
streamlit run app.py.
```
+ Visit localhost:8501 on your web browser to access the web app.

## Contributions
Contributions to this project are welcome. If you find any issues or have any suggestions for improvement, please open an issue or a pull request on this repository.

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Function to read and preprocess the data
def read_data(file_path, has_genre=True):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(':::')
            data.append(parts)
    if has_genre:
        return pd.DataFrame(data, columns=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
    else:
        return pd.DataFrame(data, columns=['ID', 'TITLE', 'DESCRIPTION'])

# Reading train and test data
train_data = read_data('train_data.txt')
test_data = read_data('test_data.txt', has_genre=False)

# Combine title and description into one feature
train_data['TEXT'] = train_data['TITLE'] + ' ' + train_data['DESCRIPTION']
test_data['TEXT'] = test_data['TITLE'] + ' ' + test_data['DESCRIPTION']

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Lowercase
    return text

# Apply cleaning
train_data['TEXT'] = train_data['TEXT'].apply(clean_text)
test_data['TEXT'] = test_data['TEXT'].apply(clean_text)

# Split train data for validation
X_train, X_val, y_train, y_val = train_test_split(train_data['TEXT'], train_data['GENRE'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

# Models
logistic_regression = LogisticRegression(max_iter=200)
decision_tree = DecisionTreeClassifier()
svm = SVC(kernel='linear')

# Pipelines
pipelines = {
    'Logistic Regression': Pipeline([('tfidf', tfidf), ('clf', logistic_regression)]),
    'Decision Tree': Pipeline([('tfidf', tfidf), ('clf', decision_tree)]),
    'SVM': Pipeline([('tfidf', tfidf), ('clf', svm)])
}

# Train and evaluate each model
for name, pipeline in pipelines.items():
    print(f'Training {name}...')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    print(f'Evaluation for {name}:\n')
    print(classification_report(y_val, y_pred))
    print('-' * 80)

# Predict on test data
for name, pipeline in pipelines.items():
    test_data[f'PREDICTED_GENRE_{name.replace(" ", "_")}'] = pipeline.predict(test_data['TEXT'])

# Save predictions for each model
test_data.to_csv('test_predictions.csv', index=False)

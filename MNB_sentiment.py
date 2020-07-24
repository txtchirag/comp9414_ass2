# Load libraries
import re
import sys

import numpy as np
import pandas as pd


from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


def predict_and_test(model, X_test_bag_of_words):
    predicted_y = model.predict(X_test_bag_of_words)
    for i in range(len(predicted_y)):
        print(test['instance_number'].iloc[i], predicted_y[i])
    # print(y_test, predicted_y)
    # print(model.predict_proba(X_test_bag_of_words))
    # print(classification_report(y_test, predicted_y))


train = pd.read_csv(sys.argv[1], sep='\t', names=['instance_number', 'tweet_text', 'sentiment'])
test = pd.read_csv(sys.argv[2], sep='\t', names=['instance_number', 'tweet_text', 'sentiment'])


def load_data():
    feature_col = ['tweet_text']
    label_col = ['sentiment']

    def pre_processing(df):
        df = df.replace(regex=' http[0-9A-Za-z:/.]* ?', value=' ')
        df = df.replace(regex='[^a-zA-Z0-9#@_$%\s]', value='')
        df = df.replace(regex='^\s+|\s+$', value='')
        df = df.replace(regex='\s+', value=' ')
        return df

    X_train = train[feature_col]  # Features
    X_train = pre_processing(X_train)
    y_train = train[label_col]  # Target variable

    X_test = test[feature_col]  # Features

    X_test = pre_processing(X_test)
    y_test = test[label_col]  # Target variable

    # return X_train,X_test,y_train,y_test
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()

X_train = X_train.values.flatten().tolist()
X_test = X_test.values.flatten().tolist()

count = CountVectorizer(token_pattern='[a-zA-Z0-9_@#$%]{2,}', lowercase=False)
X_train_bag_of_words = count.fit_transform(X_train, y=y_train)

# transform the test data into bag of words creaed with fit_transform
X_test_bag_of_words = count.transform(X_test)

# if random_state id not set. the feaures are randomised, therefore tree may be different each time
#print("----mnb")
clf = MultinomialNB()
model = clf.fit(X_train_bag_of_words, y_train.values.ravel())
predict_and_test(model, X_test_bag_of_words)



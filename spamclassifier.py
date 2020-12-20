# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:28:02 2020

@author: Prakash Gupta
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names = ['label','message'], encoding='latin-1')

messages.head()

ps = PorterStemmer()
wordnet = WordNetLemmatizer()

corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(corpus).toarray()

pickle.dump(cv, open('cv-transform.pkl', 'wb'))

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
spam_detect_model_cv = nb.fit(X_train, y_train)

pickle.dump(spam_detect_model_cv, open('model_cv.pkl', 'wb'))

y_pred=spam_detect_model_cv.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test,y_pred)
score


#     Stem
#     all features
#     accuracy = 0.979372197309417
#     cm = array([[940,  15],
#            [  8, 152]], dtype=int64)

#     stem
#     2500 features
#     accuracy = 0.9838565022421525
#     cm = array([[945,  10],
#            [  8, 152]], dtype=int64)

#     stem
#     200 features
#     accuracy = 0.9847533632286996
#     cm = array([[946,   9],
#            [  8, 152]], dtype=int64)

# ## Using Lemmatizer and Tfidf Vectorizer

corpus_lm = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus_lm.append(review)


# Creating the TFIDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=2500)
X = tfidf.fit_transform(corpus_lm).toarray()

pickle.dump(tfidf, open('tfidf-transform.pkl', 'wb'))

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training model using Naive bayes classifier

nb = MultinomialNB()
spam_detect_model_tfidf = nb.fit(X_train, y_train)

pickle.dump(spam_detect_model_tfidf, open('model_tfidf.pkl', 'wb'))

y_pred=spam_detect_model_tfidf.predict(X_test)


cm = confusion_matrix(y_test,y_pred)
cm

score = accuracy_score(y_test,y_pred)
score


# ## Conclusion

# CountVectorizer with Porter Stemmer and Max_features = 2000 with MultinomialNB gave us the highest accuracy






# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:28:02 2020

@author: Prakash Gupta
"""

from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

filename = 'model_cv.pkl'
transform_name = 'cv-transform.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open(transform_name, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)

if __name__ == "__main__":
    app.run(debug=True)
    
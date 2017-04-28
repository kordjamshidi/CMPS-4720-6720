#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Katie Roberts
Machine Learning
First Simple Runnable Model Assignment
3.6.17
"""

import pandas as pd
#from sklearn.utils import shuffle
import nltk
import csv
from sklearn.feature_extraction import DictVectorizer
#import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pickle

#loading and splitting data into train and test sets
data = pd.read_csv('Documents/sarcasmData.csv')
#data = shuffle(data)
test = data.loc[(0.75*len(data)):]
train = data.loc[:(0.75*len(data))]
train.to_csv('Documents/train.csv')
test.to_csv('Documents/test.csv')                

def features(path): #will eventually include all features (sentiment, capitalization, POS, etc.), not just ngrams.

    features = {}
    findGrams(features, path)
    return features

def findGrams(path): #finds unigrams and bigrams in the data

    tokens = []
    bigrams = []
    with open('{}'.format(path), 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            text = row[4]
        if text.isalpha():
            tokenized = nltk.word_tokenize(text)
            tokens.append(tokenized)
        porter = nltk.PorterStemmer()
        tokens = [porter.stem(t.lower()) for t in tokens]
        bigrams = nltk.bigrams(tokens)
        bigrams = [tup[0]+' '+tup[1] for tup in bigrams]
        grams = tokens + bigrams
    
        for t in grams:
            features['contains(%s)' % t] = 1.0
                     
#transforming the features into a vector to use in classification

vectorizer = DictVectorizer()
featureVector = vectorizer.fit_transform(features)
filename = 'vectorDictionary.p'
fileobject = open(filename, 'wb')
pickle.dump(vectorizer, fileobject)
fileobject.close()


#creating feature vectors for train and test sets    
trainFeatures = features('Documents/train.csv')
testFeatures = features('Documents/test.csv')

#creating the target values for train and test sets
trainTargets = []        
with open('Documents/train.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        text = row[2]
        trainTargets.append(text)

testTargets = []       
with open('Documents/train.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        text = row[2]
        testTargets.append(text)

#enumerating values for sarcastic and not sarcastic labels
for value in trainTargets:
    if value == 'sarc':
        value.replace('sarc', 1)
    else:
        value.replace('notsarc', 0)

#classifying
classifier = LinearSVC()
classifier.fit(trainFeatures, trainTargets)

#saving
filename = 'classifier.p'
fileobject = open(filename, 'wb')
pickle.dump(classifier, fileobject)
fileobject.close()

#predicting
classes = ['sarc', 'notsarc']
prediction = classifier.predict(testFeatures)
print classification_report(testTargets, prediction, target_names=classes)


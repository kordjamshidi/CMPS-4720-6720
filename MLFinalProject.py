#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Katie Roberts
Machine Learning Final Project
"""
import csv
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction import DictVectorizer
import random
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

"""Data Formatting"""
#importing and putting data into lists
reader = csv.reader(open('Desktop/sarcasm.csv'))
labels = []
tweets = []
for row in reader:
    labels.append(row[0])
    tweets.append(row[1])

#dropping header
labels = labels[1:]
tweets = tweets[1:]

#putting texts and labels together and shuffling them
labeled = ([(tweet, label) for tweet, label in zip(tweets, labels)])
random.shuffle(labeled)


"""Master Feature Function"""
#function to create a feature set from all feature extractors

def allFeatures(tweet):
    features = {}
    gramsFeatures(features, tweet)
    capitalizationFeatures(features, tweet)
    emoticonFeatures(features, tweet)
    punctuationFeatures(features, tweet)
    return features
    

"""Unigrams and Bigrams Feature Extractor"""
#function to extract unigrams and bigrams from each tweet

def gramsFeatures(features, tweet):
    unigrams = word_tokenize(tweet.lower())
    bigrams = nltk.bigrams(unigrams)
    bigrams = [bigram[0]+' '+bigram[1] for bigram in bigrams]
    allgrams = unigrams + bigrams
    for gram in allgrams:
        features["has({})".format(gram)] = (gram in tweet)
    
"""Capitalization Feature Extractor"""
#function to extract the number of capitalized letters per tweet

def capitalizationFeatures(features, tweet):
    counter = 0
    for i in range(len(tweet)):
        if tweet[i].isupper():
            counter += 1
    features['Number of capitalized letters'] = counter   

"""Exclamation/Question Mark Feature Extractor"""
#function to extract counts of and possession of punctuation  marks per tweet

def punctuationFeatures(features, tweet):
    exclamCounter = 0
    exclamThreshold = 3
    questionCounter = 0
    questionThreshold = 3
    for item in range(len(tweet)):
        if tweet[item] == '!':
            exclamCounter += 1
        if tweet[item] == '?':
            questionCounter += 1
    features["Exclamation Points"] = int(exclamCounter>=exclamThreshold)
    features["Question Marks"] = int(questionCounter>=questionThreshold)   
    
"""Emoticon Feature Extractor"""

def emoticonFeatures(features, tweet):
    types = ['emoticonXAgree',
 'emoticonXAngel',
 'emoticonXBanghead',
 'emoticonXBouncer',
 'emoticonXBye',
 'emoticonXCensored',
 'emoticonXChicken',
 'emoticonXClown',
 'emoticonXConfused',
 'emoticonXCry',
 'emoticonXDonno',
 'emoticonXEmbarrassed',
 'emoticonXFrazzled',
 'emoticonXGood',
 'emoticonXHoho',
 'emoticonXIc',
 'emoticonXIdea',
 'emoticonXKill',
 'emoticonXLove',
 'emoticonXRolleyes',
 'emoticonXSmilie',
 'emoticonXWow'] 
    for emoticon in types:
        features["count({})".format(emoticon)] = tweet.count(emoticon)
        features["has({})".format(emoticon)] = (emoticon in tweet)
        
    
"""Classification Using All Features"""

#creating master feature set
featureSet = [allFeatures(tweet) for (tweet, label) in labeled]
targetSet = [label for (tweet, label) in labeled]
updatedTargetSet = []
for label in targetSet:
    if label == 'sarc':
        updatedTargetSet.append(1)
    else:
        updatedTargetSet.append(0)
targetSet = updatedTargetSet


#creating train and test feature sets
trainFeatures = featureSet[:int(.75*len(featureSet))]
testFeatures = featureSet[int(.75*len(featureSet)):]

#creating train and test target sets with binary values
trainTargets = targetSet[:int(.75*len(targetSet))]
testTargets = targetSet[int(.75*len(targetSet)):]                           

#vectorizing feature sets for use in classification
vectorizer = DictVectorizer()                         
trainFeatures = vectorizer.fit_transform(trainFeatures)
testFeatures = vectorizer.transform(testFeatures)

#classifying using SVM
SVMclassifier = LinearSVC()
SVMclassifier.fit(trainFeatures, trainTargets)
predicted = SVMclassifier.predict(testFeatures)
target_names = ['notsarc', 'sarc']
print(classification_report(testTargets, predicted, target_names=target_names))

#classifying using Naive Bayes
NBclassifier = MultinomialNB()
NBclassifier.fit(trainFeatures, trainTargets)
predicted = NBclassifier.predict(testFeatures)
target_names = ['notsarc', 'sarc']
print(classification_report(testTargets, predicted, target_names=target_names))



""" Evaluation """
#Function to extract the most informative features from each classifier
def mostInformativeFeatures(vectorizer, classifier, n=20):
    featureNames = vectorizer.get_feature_names()
    coefsAndFns = sorted(zip(classifier.coef_[0], featureNames))
    top = zip(coefsAndFns[:n], coefsAndFns[:-(n + 1):-1])
    for (coef1, fn1), (coef2, fn2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef1, fn1, coef2, fn2)
        
print("Most informative features for SVM: ")
mostInformativeFeatures(vectorizer, SVMclassifier)
print("Most informative features for Naive Bayes: ")
mostInformativeFeatures(vectorizer, NBclassifier)


#Implementing 5-fold Cross Validation
SVMscores = cross_val_score(SVMclassifier, testFeatures, testTargets, cv=5, scoring='f1_macro')
NBscores = cross_val_score(NBclassifier, testFeatures, testTargets, cv=5, scoring='f1_macro')
print("SVM scores: ")
print(SVMscores)
print("Naive Bayes Scores: ")
print(NBscores)
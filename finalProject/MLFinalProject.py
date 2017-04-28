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
import loadSent
import numpy as np

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
    sentimentFeatures(features, tweet)
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
"""Sentiment Features"""

#evaluates sentiments for each sentence as well as contrasts in sentiment throughout the sentence

def sentimentFeatures(features, tweet):
    sentiments = loadSent.loadSent()
    tokens = nltk.word_tokenize(tweet)
    tokens = [(t.lower()) for t in tokens] 
    
    #mean sentiment for overall sentence
    mean_sentiment = sentiments.scoreSentence(tokens)
    features['Positive sentiment'] = mean_sentiment[0]
    features['Negative sentiment'] = mean_sentiment[1]
    features['Sentiment'] = mean_sentiment[0]-mean_sentiment[1]
    
  
    #Splitting sentence in half
    if len(tokens)==1:
        tokens+=['.']
    fHalf = tokens[0:len(tokens)/2]
    sHalf = tokens[len(tokens)/2:]
    
    
    mean_sentiment_f = sentiments.scoreSentence(fHalf)
    features['Positive sentiment first half'] = mean_sentiment_f[0]
    features['Negative sentiment first half'] = mean_sentiment_f[1]
    features['Sentiment first half'] = mean_sentiment_f[0]-mean_sentiment_f[1]
    
    mean_sentiment_s = sentiments.scoreSentence(sHalf)
    features['Positive sentiment second half'] = mean_sentiment_s[0]
    features['Negative sentiment second half'] = mean_sentiment_s[1]
    features['Sentiment second half'] = mean_sentiment_s[0]-mean_sentiment_s[1]
    
    features['Sentiment contrast 2'] = np.abs(features['Sentiment 1/2']-features['Sentiment 2/2'])

  
    #Split sentence into three
    if len(tokens)==2:
        tokens+=['.']
    fHalf = tokens[0:len(tokens)/3]
    sHalf = tokens[len(tokens)/3:2*len(tokens)/3]
    tHalf = tokens[2*len(tokens)/3:]
    
    meanSentiment_f = sentiments.scoreSentence(fHalf)
    features['Positive sentiment first third'] = meanSentiment_f[0]
    features['Negative sentiment first third'] = meanSentiment_f[1]
    features['Sentiment first third'] = meanSentiment_f[0]-meanSentiment_f[1]
    
    meanSentiment_s = sentiments.scoreSentence(sHalf)
    features['Positive sentiment second third'] = meanSentiment_s[0]
    features['Negative sentiment second third'] = meanSentiment_s[1]
    features['Sentiment second third'] = meanSentiment_s[0]-meanSentiment_s[1]
    
    meanSentiment_t = sentiments.scoreSentence(tHalf)
    features['Positive sentiment third third'] = meanSentiment_t[0]
    features['Negative sentiment third third'] = meanSentiment_t[1]
    features['Sentiment third'] = meanSentiment_t[0]-meanSentiment_t[1]
    
    features['Sentiment contrast 3'] = np.abs(features['Sentiment 1/3']-features['Sentiment 3/3'])
            
    
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
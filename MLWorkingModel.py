"""
Katie Roberts
Machine Learning
Reasonably Good Working Model Assignment
"""
import csv
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction import DictVectorizer
import random
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

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
    return features
    

"""Unigrams and Bigrams Feature Extractor"""
#function to extract unigrams and bigrams from each tweet
def gramsFeatures(features, tweet):
    unigrams = word_tokenize(tweet.lower())
    bigrams = nltk.bigrams(unigrams)
    bigrams = [bigram[0]+' '+bigram[1] for bigram in bigrams]
    allgrams = unigrams + bigrams
    for a in allgrams:
        features['contains "{}"'.format(a)] = 1.0
    
                 

"""Classification Using Unigrams and Bigrams Features"""

#creating master feature set
featureSet = [gramsFeatures(tweet) for (tweet, label) in labeled]
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
classifier = LinearSVC()
classifier.fit(trainFeatures, trainTargets)
predicted = classifier.predict(testFeatures)
target_names = ['notsarc', 'sarc']
print(classification_report(testTargets, predicted, target_names=target_names))
#average F1=0.72, notsarc=.71, sarc=.73

#classifying using Naive Bayes
classifier = MultinomialNB()
classifier.fit(trainFeatures, trainTargets)
predicted = classifier.predict(testFeatures)
target_names = ['notsarc', 'sarc']
print(classification_report(testTargets, predicted, target_names=target_names))
#average F1=.71, notsarc=.74, sarc=.69


"""Capitalization Feature Extractor"""
#function to extract the number of capitalized letters per tweet

def capitalizationFeatures(features, tweet):
    counter = 0
    for i in range(len(tweet)):
        if tweet[i].isupper():
            counter += 1
    features['Number of capitalized letters'] = counter


"""Classification Using Capitalization + unigrams + bigrams"""

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
classifier = LinearSVC()
classifier.fit(trainFeatures, trainTargets)
predicted = classifier.predict(testFeatures)
target_names = ['notsarc', 'sarc']
print(classification_report(testTargets, predicted, target_names=target_names))
#averageF1=.74, notsarc=.73, sarc = .74

#classifying using Naive Bayes
classifier = MultinomialNB()
classifier.fit(trainFeatures, trainTargets)
predicted = classifier.predict(testFeatures)
target_names = ['notsarc', 'sarc']
print(classification_report(testTargets, predicted, target_names=target_names))
#average F1=.73, notsarc=.76, sarc=.70












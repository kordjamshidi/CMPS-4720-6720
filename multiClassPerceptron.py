# -*- coding: utf-8 -*-
"""
Katie Roberts
Machine Learning
Programming Assignment 1 - Multi Class Perceptron
"""

import numpy as np
import random

#formatted Iris Data

#lists iris classes
irisClasses = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

#Lists features provided: sepLen = Sepal Length, sepWid = Sepal Width, petLen = Petal Length, petWid = Petal Width
irisFeatureList = ['sepLen', 'sepWid', 'petLen', 'petWid']

#Formats class and feature data to be read in
irisFeatureData = [('Iris-setosa', {'sepLen': 5.1, 'sepWid': 3.5, 'petLen': 1.4, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 4.9, 'sepWid': 3.0, 'petLen': 1.4, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 4.7, 'sepWid': 3.2, 'petLen': 1.3, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 4.6, 'sepWid': 3.1, 'petLen': 1.5, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.0, 'sepWid': 3.6, 'petLen': 1.4, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.4, 'sepWid': 3.9, 'petLen': 1.7, 'petWid': 0.4}), ('Iris-setosa', {'sepLen': 4.6, 'sepWid': 3.4, 'petLen': 1.4, 'petWid': 0.3}), ('Iris-setosa', {'sepLen': 5.0, 'sepWid': 3.4, 'petLen': 1.5, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 4.4, 'sepWid': 2.9, 'petLen': 1.4, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 4.9, 'sepWid': 3.1, 'petLen': 1.5, 'petWid': 0.1}), ('Iris-setosa', {'sepLen': 5.4, 'sepWid': 3.7, 'petLen': 1.5, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 4.8, 'sepWid': 3.4, 'petLen': 1.6, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 4.8, 'sepWid': 3.0, 'petLen': 1.1, 'petWid': 0.1}), ('Iris-setosa', {'sepLen': 4.3, 'sepWid': 3.0, 'petLen': 1.1, 'petWid': 0.1}), ('Iris-setosa', {'sepLen': 5.8, 'sepWid': 4.0, 'petLen': 1.2, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.7, 'sepWid': 4.4, 'petLen': 1.5, 'petWid': 0.4}), ('Iris-setosa', {'sepLen': 5.4, 'sepWid': 3.9, 'petLen': 1.3, 'petWid': 0.4}), ('Iris-setosa', {'sepLen': 5.1, 'sepWid': 3.5, 'petLen': 1.4, 'petWid': 0.3}), ('Iris-setosa', {'sepLen': 5.7, 'sepWid': 3.8, 'petLen': 1.7, 'petWid': 0.3}), ('Iris-setosa', {'sepLen': 5.1, 'sepWid': 3.8, 'petLen': 1.5, 'petWid': 0.3}), ('Iris-setosa', {'sepLen': 5.4, 'sepWid': 3.4, 'petLen': 1.7, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.1, 'sepWid': 3.7, 'petLen': 1.5, 'petWid': 0.4}), ('Iris-setosa', {'sepLen': 4.6, 'sepWid': 3.6, 'petLen': 1.0, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.1, 'sepWid': 3.3, 'petLen': 1.7, 'petWid': 0.5}), ('Iris-setosa', {'sepLen': 4.8, 'sepWid': 3.4, 'petLen': 1.9, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.0, 'sepWid': 3.0, 'petLen': 1.6, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.0, 'sepWid': 3.4, 'petLen': 1.6, 'petWid': 0.4}), ('Iris-setosa', {'sepLen': 5.2, 'sepWid': 3.5, 'petLen': 1.5, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.2, 'sepWid': 3.4, 'petLen': 1.4, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 4.7, 'sepWid': 3.2, 'petLen': 1.6, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 4.8, 'sepWid': 3.1, 'petLen': 1.6, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.4, 'sepWid': 3.4, 'petLen': 1.5, 'petWid': 0.4}), ('Iris-setosa', {'sepLen': 5.2, 'sepWid': 4.1, 'petLen': 1.5, 'petWid': 0.1}), ('Iris-setosa', {'sepLen': 5.5, 'sepWid': 4.2, 'petLen': 1.4, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 4.9, 'sepWid': 3.1, 'petLen': 1.5, 'petWid': 0.1}), ('Iris-setosa', {'sepLen': 5.0, 'sepWid': 3.2, 'petLen': 1.2, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.5, 'sepWid': 3.5, 'petLen': 1.3, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 4.9, 'sepWid': 3.1, 'petLen': 1.5, 'petWid': 0.1}), ('Iris-setosa', {'sepLen': 4.4, 'sepWid': 3.0, 'petLen': 1.3, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.1, 'sepWid': 3.4, 'petLen': 1.5, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.0, 'sepWid': 3.5, 'petLen': 1.3, 'petWid': 0.3}), ('Iris-setosa', {'sepLen': 4.5, 'sepWid': 2.3, 'petLen': 1.3, 'petWid': 0.3}), ('Iris-setosa', {'sepLen': 4.4, 'sepWid': 3.2, 'petLen': 1.3, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.0, 'sepWid': 3.5, 'petLen': 1.6, 'petWid': 0.6}), ('Iris-setosa', {'sepLen': 5.1, 'sepWid': 3.8, 'petLen': 1.9, 'petWid': 0.4}), ('Iris-setosa', {'sepLen': 4.8, 'sepWid': 3.0, 'petLen': 1.4, 'petWid': 0.3}), ('Iris-setosa', {'sepLen': 5.1, 'sepWid': 3.8, 'petLen': 1.6, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 4.6, 'sepWid': 3.2, 'petLen': 1.4, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.3, 'sepWid': 3.7, 'petLen': 1.5, 'petWid': 0.2}), ('Iris-setosa', {'sepLen': 5.0, 'sepWid': 3.3, 'petLen': 1.4, 'petWid': 0.2}), ('Iris-versicolor', {'sepLen': 7.0, 'sepWid': 3.2, 'petLen': 4.7, 'petWid': 1.4}), ('Iris-versicolor', {'sepLen': 6.4, 'sepWid': 3.2, 'petLen': 4.5, 'petWid': 1.5}), ('Iris-versicolor', {'sepLen': 6.9, 'sepWid': 3.1, 'petLen': 4.9, 'petWid': 1.5}), ('Iris-versicolor', {'sepLen': 5.5, 'sepWid': 2.3, 'petLen': 4.0, 'petWid': 1.3}), ('Iris-versicolor', {'sepLen': 6.5, 'sepWid': 2.8, 'petLen': 4.6, 'petWid': 1.5}), ('Iris-versicolor', {'sepLen': 5.7, 'sepWid': 2.8, 'petLen': 4.5, 'petWid': 1.3}), ('Iris-versicolor', {'sepLen': 6.3, 'sepWid': 3.3, 'petLen': 4.7, 'petWid': 1.6}), ('Iris-versicolor', {'sepLen': 4.9, 'sepWid': 2.4, 'petLen': 3.3, 'petWid': 1.0}), ('Iris-versicolor', {'sepLen': 6.6, 'sepWid': 2.9, 'petLen': 4.6, 'petWid': 1.3}), ('Iris-versicolor', {'sepLen': 5.2, 'sepWid': 2.7, 'petLen': 3.9, 'petWid': 1.4}), ('Iris-versicolor', {'sepLen': 5.0, 'sepWid': 2.0, 'petLen': 3.5, 'petWid': 1.0}), ('Iris-versicolor', {'sepLen': 5.9, 'sepWid': 3.0, 'petLen': 4.2, 'petWid': 1.5}), ('Iris-versicolor', {'sepLen': 6.0, 'sepWid': 2.2, 'petLen': 4.0, 'petWid': 1.0}), ('Iris-versicolor', {'sepLen': 6.1, 'sepWid': 2.9, 'petLen': 4.7, 'petWid': 1.4}), ('Iris-versicolor', {'sepLen': 5.6, 'sepWid': 2.9, 'petLen': 3.6, 'petWid': 1.3}), ('Iris-versicolor', {'sepLen': 6.7, 'sepWid': 3.1, 'petLen': 4.1, 'petWid': 1.4}), ('Iris-versicolor', {'sepLen': 5.6, 'sepWid': 3.0, 'petLen': 4.5, 'petWid': 1.5}), ('Iris-versicolor', {'sepLen': 5.8, 'sepWid': 2.7, 'petLen': 4.1, 'petWid': 1.0}), ('Iris-versicolor', {'sepLen': 6.2, 'sepWid': 2.2, 'petLen': 4.5, 'petWid': 1.5}), ('Iris-versicolor', {'sepLen': 5.6, 'sepWid': 2.5, 'petLen': 3.9, 'petWid': 1.1}), ('Iris-versicolor', {'sepLen': 5.9, 'sepWid': 3.2, 'petLen': 4.8, 'petWid': 1.8}), ('Iris-versicolor', {'sepLen': 6.1, 'sepWid': 2.8, 'petLen': 4.0, 'petWid': 1.3}), ('Iris-versicolor', {'sepLen': 6.3, 'sepWid': 2.5, 'petLen': 4.9, 'petWid': 1.5}), ('Iris-versicolor', {'sepLen': 6.1, 'sepWid': 2.8, 'petLen': 4.7, 'petWid': 1.2}), ('Iris-versicolor', {'sepLen': 6.4, 'sepWid': 2.9, 'petLen': 4.3, 'petWid': 1.3}), ('Iris-versicolor', {'sepLen': 6.6, 'sepWid': 3.0, 'petLen': 4.4, 'petWid': 1.4}), ('Iris-versicolor', {'sepLen': 6.8, 'sepWid': 2.8, 'petLen': 4.8, 'petWid': 1.4}), ('Iris-versicolor', {'sepLen': 6.7, 'sepWid': 3.0, 'petLen': 5.0, 'petWid': 1.7}), ('Iris-versicolor', {'sepLen': 6.0, 'sepWid': 2.9, 'petLen': 4.5, 'petWid': 1.5}), ('Iris-versicolor', {'sepLen': 5.7, 'sepWid': 2.6, 'petLen': 3.5, 'petWid': 1.0}), ('Iris-versicolor', {'sepLen': 5.5, 'sepWid': 2.4, 'petLen': 3.8, 'petWid': 1.1}), ('Iris-versicolor', {'sepLen': 5.5, 'sepWid': 2.4, 'petLen': 3.7, 'petWid': 1.0}), ('Iris-versicolor', {'sepLen': 5.8, 'sepWid': 2.7, 'petLen': 3.9, 'petWid': 1.2}), ('Iris-versicolor', {'sepLen': 6.0, 'sepWid': 2.7, 'petLen': 5.1, 'petWid': 1.6}), ('Iris-versicolor', {'sepLen': 5.4, 'sepWid': 3.0, 'petLen': 4.5, 'petWid': 1.5}), ('Iris-versicolor', {'sepLen': 6.0, 'sepWid': 3.4, 'petLen': 4.5, 'petWid': 1.6}), ('Iris-versicolor', {'sepLen': 6.7, 'sepWid': 3.1, 'petLen': 4.7, 'petWid': 1.5}), ('Iris-versicolor', {'sepLen': 6.3, 'sepWid': 2.3, 'petLen': 4.4, 'petWid': 1.3}), ('Iris-versicolor', {'sepLen': 5.6, 'sepWid': 3.0, 'petLen': 4.1, 'petWid': 1.3}), ('Iris-versicolor', {'sepLen': 5.5, 'sepWid': 2.5, 'petLen': 4.0, 'petWid': 1.3}), ('Iris-versicolor', {'sepLen': 5.5, 'sepWid': 2.6, 'petLen': 4.4, 'petWid': 1.2}), ('Iris-versicolor', {'sepLen': 6.1, 'sepWid': 3.0, 'petLen': 4.6, 'petWid': 1.4}), ('Iris-versicolor', {'sepLen': 5.8, 'sepWid': 2.6, 'petLen': 4.0, 'petWid': 1.2}), ('Iris-versicolor', {'sepLen': 5.0, 'sepWid': 2.3, 'petLen': 3.3, 'petWid': 1.0}), ('Iris-versicolor', {'sepLen': 5.6, 'sepWid': 2.7, 'petLen': 4.2, 'petWid': 1.3}), ('Iris-versicolor', {'sepLen': 5.7, 'sepWid': 3.0, 'petLen': 4.2, 'petWid': 1.2}), ('Iris-versicolor', {'sepLen': 5.7, 'sepWid': 2.9, 'petLen': 4.2, 'petWid': 1.3}), ('Iris-versicolor', {'sepLen': 6.2, 'sepWid': 2.9, 'petLen': 4.3, 'petWid': 1.3}), ('Iris-versicolor', {'sepLen': 5.1, 'sepWid': 2.5, 'petLen': 3.0, 'petWid': 1.1}), ('Iris-versicolor', {'sepLen': 5.7, 'sepWid': 2.8, 'petLen': 4.1, 'petWid': 1.3}), ('Iris-virginica', {'sepLen': 6.3, 'sepWid': 3.3, 'petLen': 6.0, 'petWid': 2.5}), ('Iris-virginica', {'sepLen': 5.8, 'sepWid': 2.7, 'petLen': 5.1, 'petWid': 1.9}), ('Iris-virginica', {'sepLen': 7.1, 'sepWid': 3.0, 'petLen': 5.9, 'petWid': 2.1}), ('Iris-virginica', {'sepLen': 6.3, 'sepWid': 2.9, 'petLen': 5.6, 'petWid': 1.8}), ('Iris-virginica', {'sepLen': 6.5, 'sepWid': 3.0, 'petLen': 5.8, 'petWid': 2.2}), ('Iris-virginica', {'sepLen': 7.6, 'sepWid': 3.0, 'petLen': 6.6, 'petWid': 2.1}), ('Iris-virginica', {'sepLen': 4.9, 'sepWid': 2.5, 'petLen': 4.5, 'petWid': 1.7}), ('Iris-virginica', {'sepLen': 7.3, 'sepWid': 2.9, 'petLen': 6.3, 'petWid': 1.8}), ('Iris-virginica', {'sepLen': 6.7, 'sepWid': 2.5, 'petLen': 5.8, 'petWid': 1.8}), ('Iris-virginica', {'sepLen': 7.2, 'sepWid': 3.6, 'petLen': 6.1, 'petWid': 2.5}), ('Iris-virginica', {'sepLen': 6.5, 'sepWid': 3.2, 'petLen': 5.1, 'petWid': 2.0}), ('Iris-virginica', {'sepLen': 6.4, 'sepWid': 2.7, 'petLen': 5.3, 'petWid': 1.9}), ('Iris-virginica', {'sepLen': 6.8, 'sepWid': 3.0, 'petLen': 5.5, 'petWid': 2.1}), ('Iris-virginica', {'sepLen': 5.7, 'sepWid': 2.5, 'petLen': 5.0, 'petWid': 2.0}), ('Iris-virginica', {'sepLen': 5.8, 'sepWid': 2.8, 'petLen': 5.1, 'petWid': 2.4}), ('Iris-virginica', {'sepLen': 6.4, 'sepWid': 3.2, 'petLen': 5.3, 'petWid': 2.3}), ('Iris-virginica', {'sepLen': 6.5, 'sepWid': 3.0, 'petLen': 5.5, 'petWid': 1.8}), ('Iris-virginica', {'sepLen': 7.7, 'sepWid': 3.8, 'petLen': 6.7, 'petWid': 2.2}), ('Iris-virginica', {'sepLen': 7.7, 'sepWid': 2.6, 'petLen': 6.9, 'petWid': 2.3}), ('Iris-virginica', {'sepLen': 6.0, 'sepWid': 2.2, 'petLen': 5.0, 'petWid': 1.5}), ('Iris-virginica', {'sepLen': 6.9, 'sepWid': 3.2, 'petLen': 5.7, 'petWid': 2.3}), ('Iris-virginica', {'sepLen': 5.6, 'sepWid': 2.8, 'petLen': 4.9, 'petWid': 2.0}), ('Iris-virginica', {'sepLen': 7.7, 'sepWid': 3.8, 'petLen': 6.7, 'petWid': 2.0}), ('Iris-virginica', {'sepLen': 6.3, 'sepWid': 2.7, 'petLen': 4.9, 'petWid': 1.8}), ('Iris-virginica', {'sepLen': 6.7, 'sepWid': 3.3, 'petLen': 5.7, 'petWid': 2.1}), ('Iris-virginica', {'sepLen': 7.2, 'sepWid': 3.2, 'petLen': 6.0, 'petWid': 1.8}), ('Iris-virginica', {'sepLen': 6.2, 'sepWid': 2.8, 'petLen': 4.8, 'petWid': 1.8}), ('Iris-virginica', {'sepLen': 6.1, 'sepWid': 3.0, 'petLen': 4.9, 'petWid': 1.8}), ('Iris-virginica', {'sepLen': 6.4, 'sepWid': 2.8, 'petLen': 5.6, 'petWid': 2.1}), ('Iris-virginica', {'sepLen': 7.2, 'sepWid': 3.0, 'petLen': 5.8, 'petWid': 1.6}), ('Iris-virginica', {'sepLen': 7.4, 'sepWid': 2.8, 'petLen': 6.1, 'petWid': 1.9}), ('Iris-virginica', {'sepLen': 7.9, 'sepWid': 3.8, 'petLen': 6.4, 'petWid': 2.0}), ('Iris-virginica', {'sepLen': 6.4, 'sepWid': 2.8, 'petLen': 5.6, 'petWid': 2.2}), ('Iris-virginica', {'sepLen': 6.3, 'sepWid': 2.8, 'petLen': 5.1, 'petWid': 1.5}), ('Iris-virginica', {'sepLen': 6.1, 'sepWid': 2.6, 'petLen': 5.6, 'petWid': 1.4}), ('Iris-virginica', {'sepLen': 7.7, 'sepWid': 3.0, 'petLen': 6.1, 'petWid': 2.3}), ('Iris-virginica', {'sepLen': 6.3, 'sepWid': 3.4, 'petLen': 5.6, 'petWid': 2.4}), ('Iris-virginica', {'sepLen': 6.4, 'sepWid': 3.1, 'petLen': 5.5, 'petWid': 1.8}), ('Iris-virginica', {'sepLen': 6.0, 'sepWid': 3.0, 'petLen': 4.8, 'petWid': 1.8}), ('Iris-virginica', {'sepLen': 6.9, 'sepWid': 3.1, 'petLen': 5.4, 'petWid': 2.1}), ('Iris-virginica', {'sepLen': 6.7, 'sepWid': 3.1, 'petLen': 5.6, 'petWid': 2.4}), ('Iris-virginica', {'sepLen': 6.9, 'sepWid': 3.1, 'petLen': 5.1, 'petWid': 2.3}), ('Iris-virginica', {'sepLen': 5.8, 'sepWid': 2.7, 'petLen': 5.1, 'petWid': 1.9}), ('Iris-virginica', {'sepLen': 6.8, 'sepWid': 3.2, 'petLen': 5.9, 'petWid': 2.3}), ('Iris-virginica', {'sepLen': 6.7, 'sepWid': 3.3, 'petLen': 5.7, 'petWid': 2.5}), ('Iris-virginica', {'sepLen': 6.7, 'sepWid': 3.0, 'petLen': 5.2, 'petWid': 2.3}), ('Iris-virginica', {'sepLen': 6.3, 'sepWid': 2.5, 'petLen': 5.0, 'petWid': 1.9}), ('Iris-virginica', {'sepLen': 6.5, 'sepWid': 3.0, 'petLen': 5.2, 'petWid': 2.0}), ('Iris-virginica', {'sepLen': 6.2, 'sepWid': 3.4, 'petLen': 5.4, 'petWid': 2.3}), ('Iris-virginica', {'sepLen': 5.9, 'sepWid': 3.0, 'petLen': 5.1, 'petWid': 1.8})]


BIAS = 1
RATIO = .75
ITERATIONS = 100

class multiClassPerceptron():

    def __init__(self, classes, featureList, featureData, Ratio=RATIO, iterations=ITERATIONS):
        self.classes = classes
        self.featureList = featureList
        self.featureData = featureData
        self.ratio = Ratio
        self.iterations = iterations


        self.weightVectors = {c: np.array([0 for _ in xrange(len(featureList) + 1)]) for c in self.classes}

        random.shuffle(self.featureData)
        self.trainSet = self.featureData[:int(len(self.featureData) * self.ratio)]
        self.testSet = self.featureData[int(len(self.featureData) * self.ratio):]

    def train(self):
        for _ in xrange(self.iterations):
            for category, featureDict in self.trainSet:

                featureList = [featureDict[x] for x in self.featureList]
                featureList.append(BIAS)
                featureVector = np.array(featureList)

                argMax, predictedClass = 0, self.classes[0]

                for c in self.classes:
                    current = np.dot(featureVector, self.weightVectors[c])
                    if current >= argMax:
                        argMax, predictedClass = current, c

                if category != predictedClass:
                    self.weightVectors[category] = np.add(self.weightVectors[category], featureVector,out=self.weightVectors[category], casting="unsafe")
                    self.weightVectors[predictedClass] = np.subtract(self.weightVectors[predictedClass], featureVector)

    def predict(self, featureDict):
        featureList = [featureDict[x] for x in self.featureList]
        featureList.append(BIAS)
        featureVector = np.array(featureList)

        argMax, predictedClass = 0, self.classes[0]

        for c in self.classes:
            current = np.dot(featureVector, self.weightVectors[c])
            if current >= argMax:
                argMax, predictedClass = current, c

        return predictedClass


    def accuracy(self):
        correct, incorrect = 0, 0
        for featureDict in self.testSet:
            actualClass = featureDict[0]
            predictedClass = self.predict(featureDict[1])

            if actualClass == predictedClass:
                correct += 1
            else:
                incorrect += 1

        print "Model Accuracy:", (correct * 1.0) / ((correct + incorrect) * 1.0)


#application using the iris data

#irisClassifier = multiClassPerceptron(irisClasses, irisFeatureList, irisFeatureData)

#train(irisClassifier)

#accuracy(irisClassifier)
#output

#Model Accuracy: 0.921052631579
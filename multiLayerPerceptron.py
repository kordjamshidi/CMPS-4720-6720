#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Katie Roberts
Machine Learning
Programming Assignment 2: Multi Layer Perceptron Implementation Using Theano
2.17.17
"""

from sklearn.datasets import load_iris
import numpy as np
import theano
import theano.tensor as T


#loading in iris data using sklearn
iris = load_iris()
trainX = iris.data #feature data
trainY = iris.target #actual classes


#main multilayer perceptron function, set to iterate 70,000 times
def multiLayerPerceptron(iterations=70000):
    weight1.set_value(np.random.rand(inputDim, hiddenDim) / np.sqrt(inputDim))
    bias1.set_value(np.zeros(hiddenDim))
    weight2.set_value(np.random.rand(hiddenDim, outputDim) / np.sqrt(hiddenDim))
    bias2.set_value(np.zeros(outputDim))

    for i in range(0, iterations):
        randomBatches = np.random.randint(150,size=25)
        batchX, batchY = trainX[randomBatches], trainY[randomBatches]
        gradient(batchX, batchY)

        if i % 10000 == 0:
            print("Loss after iteration {0}: {1}".format(i, findLoss(trainX, trainY)))
            print("accuracy: {}".format(accuracy(trainX)))

#assigning parameters
inputDim = trainX.shape[1]
hiddenDim = 10
epsilon = 0.01
outputDim = len(iris.target_names)

x = T.matrix('x')
y = T.lvector('y')

#creating weights and biases 
weight1 = theano.shared(np.random.randn(inputDim, hiddenDim), name='weight1')
bias1 = theano.shared(np.zeros(hiddenDim), name='bias1')
weight2 = theano.shared(np.random.randn(hiddenDim, outputDim), name='weight2')
bias2 = theano.shared(np.zeros(outputDim), name='bias2')

z1 = x.dot(weight1) + bias1
a1 = T.nnet.softmax(z1)
z2 = a1.dot(weight2) + bias2
a2 = T.nnet.softmax(z2)

loss = T.nnet.categorical_crossentropy(a2, y).mean()
prediction = T.argmax(a2, axis=1)

#Using theano you can define functions this way, so here there are several functions defined that will be used 
#in constructing the main Multi Layer Perceptron function
findLoss = theano.function([x, y], loss)
predict = theano.function([x], prediction)
accuracy = theano.function([x], T.sum(T.eq(prediction, trainY)) 

dweight2 = T.grad(loss, weight2)
dbias2 = T.grad(loss, bias2)
dweight1 = T.grad(loss, weight1)
dbias1 = T.grad(loss, bias1)

#defining the gradient step function
gradient = theano.function(
    [x, y],
    updates=((weight2, weight2 - epsilon * dweight2),
             (weight1, weight1 - epsilon * dweight1),
             (bias2, bias2 - epsilon * dbias2),
             (bias1, bias1 - epsilon * dbias1)))
"""
Example:
   
multiLayerPerceptron()

#output
Loss after iteration 0: 1.10041928568
accuracy: 50
Loss after iteration 10000: 0.325868874659
accuracy: 142
Loss after iteration 20000: 0.208783355887
accuracy: 145
Loss after iteration 30000: 0.163938130404
accuracy: 145
Loss after iteration 40000: 0.140540237284
accuracy: 145
Loss after iteration 50000: 0.124153092176
accuracy: 145
Loss after iteration 60000: 0.115427479326
accuracy: 145
"""
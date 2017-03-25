# code for training and testing data samples
#uses multilayer perceptron -- INCOMPLETE 
import glob
import os
import numpy as np
import cv2
import ntpath
import tensorflow.python.platform
import tensorflow as tf
from scipy import sparse
import requests
import csv
from PIL import Image
from sklearn import svm
import random


def splitdata(data, switchratio):
    switch = int(switchratio*np.shape(data)[1])
    train_set = data[:switch]
    test_set = data[switch:]
    return train_set, test_set

def class_split(dataset):
    classindex = -1 #class index in the attribute vector is -1
    Positive_set = []
    Negative_set = []
    for feature_vector in dataset:
        if feature_vector[-1] == 0:
            Negative_set.append(feature_vector[:classindex])
        else:
            Positive_set.append(feature_vector[:classindex])
    return Positive_set, Negative_set


def label_split(data, label_index):
    '''input a matrix, data with each row representing
       an example with a label at data[row][label_index]
       also converts labels to integers
'''
    data_array= []
    label_array= []
    for row in data:
        t = row[label_index]
        if t == 0: #negative
            t = [0]
        else: #positive
            t = [1]
        label_array.append(t)
        del row[label_index]
        data_array.append(row)
    return data_array, label_array


def multilayer_perceptron(x, weights, biases):
    '''
    parameters:
    x: data, without labels, as an array
    weights: weight vectors, as an arry 
    biases: weight vector
       matmul - matrix multiplication
       add - element wise matrix addition
       --> x.w + b
'''
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #layer_1 = tf.nn.relu(layer_1)
    
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    
    # Output layer with linear activation
    output_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return output_layer


#parameters
label_index = 4
splitRatio = .8
learning_rate = 0.1
training_epochs = 10
batch_size = 10
display_step = 50


negative_features = np.load('featurevectors_DL_neg.npy')
positive_features = np.load('featurevectors_DL_pos.npy')

#concatenate label to the end of each feature matrix
positive_labels=np.transpose(np.ones(len(positive_features))*(1))
positive_labels = positive_labels.reshape(len(positive_features),1)
positive_features = np.concatenate((positive_features, positive_labels), axis=1)

negative_labels=np.transpose(np.ones(len(negative_features))*(0))
negative_labels = negative_labels.reshape(len(negative_features),1)
negative_features = np.concatenate((negative_features, negative_labels), axis=1)


#split into training testing
data = np.vstack((negative_features, positive_features))

#randomize data
np.random.shuffle(data)
data = np.ndarray.tolist(data)


'''------------------------model begins-------------------'''


# Network Parameters
n_hidden_1 = 100 # 1st layer number of nodes
n_hidden_2 = 100 # 2nd layer number of nodes
n_input = len(negative_features[0]) #  data input size (length of feature vector for each data point)
n_classes = 1 # binary
train_data, test_data = splitdata(data, splitRatio)

#split training data into labels and test/train
train_data, train_labels= label_split(train_data, label_index)
test_data, test_labels= label_split(test_data, label_index)


# tensorflow Graph input
x = tf.placeholder("float", [None, n_input]) #initializing data placeholder
y = tf.placeholder("float", [None, n_classes]) #initializind label placeholder

# Initializing layers weights & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model, with random weights/biases and empty x
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_data)/batch_size)
        
        # Loop over all batches
        for i in range(total_batch):
            #defining where the batch is coming from
            start = i*batch_size
            end = i*batch_size+batch_size
            if end> len(train_data):
                end = len(train_data)
                
            batch_x = train_data[start:end]
            batch_y = train_labels[start:end]

            # Run optimization op (backpropogation) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            # Compute average loss
            avg_cost += c / total_batch
            
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Done.")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    print(sess.run(accuracy, feed_dict={x: test_data, y: test_labels}))



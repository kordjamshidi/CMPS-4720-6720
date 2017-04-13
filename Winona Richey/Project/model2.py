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

def test_divide(features,names, test_image_index, hold_x_out):
        train_features = []
        train_names = []
        test_features = []
        test_names = []
        #define the names you're taking as test set
        #iterate through image indices test_image_index+[0..hold_x_out]
        image_names = []
        for x in range(hold_x_out):
                #location of filename dependent on the # of digits in the index
                image_names.append("transfer_" + str((test_image_index+x)%25))

        print "Testing with hold out of whole image number: "
        print image_names

        for x in range(len(features)):
                feature = features[x]
                name = names[x]
                this_name = name[:(9+len(str(test_image_index)))]                                        
                if this_name in image_names:
                        test_features.append(feature)
                        test_names.append(name)
                else:
                        train_features.append(feature)
                        train_names.append(name)
        return train_features, train_names, test_features, test_names
        
def get_labels(features,names):
    '''
    parameters: features: feature vector np array
                names: array of image titles as a strings; names[x] = image name for features[x]
    returns: numpy array of labels, in the same order as features and string names
             where 
    
'''
    labels = []
    for x in range(len(features)):
        feature = features[x]
        name = names[x]
        if "pos" in name:
            labels.append([1,0])
        else: #negative
            labels.append([0,1])
    return np.asarray(labels)

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
learning_rate = 0.1
training_epochs = 150
batch_size = 10
display_step = 50

features =np.load('cell_featurevectors_DL.npy')
names = np.load('cell_filenames.npy')

newfile = 'MLP_cell_results.csv' #where results will be printed
'''
f8 = open(newfile, 'wb')
fw8 = csv.writer(f8)
fw8.writerow(['Test Set','Kernel', 'deg', 'Negative Accuracy(Specificity)', 'Positive Accuracy(Sensitivity)', 'Total Accuracy'])
'''
hold_x_out = 1 #the number of images to hold out for each test (defines the training set)
x = 1 #which transfer to hold-out; will be changed in a loop eventually


#split into training and testing sets
train_data, train_names, test_data, test_names = test_divide(features,names, x,hold_x_out)

#convert names (Strings) into labels (matrices)
train_labels = get_labels(train_data, train_names)
test_labels = get_labels(test_data, test_names)



'''------------------------model begins-------------------'''


# Network Parameters
n_hidden_1 = 400 # 1st layer number of nodes
n_hidden_2 = 400 # 2nd layer number of nodes
n_input = len(train_data[0]) #  data input size (length of feature vector for each data point)
n_classes = 2 # number of labels

# tensorflow Graph input
x = tf.placeholder("float", [None, n_input]) #initializing data placeholder
y = tf.placeholder("float", [None, n_classes]) #initializind label placeholder

# Initializing layers weights & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
#need a bias for every layer
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
            if end > len(train_data):
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



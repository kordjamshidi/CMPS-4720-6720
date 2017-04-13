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
        
def label_divide(features,names):
        pos_features = []
        pos_names = []
        neg_features = []
        neg_names = []
        for x in range(len(features)):
                feature = features[x]
                name = names[x]
                if "pos" in name:
                        pos_features.append(feature)
                        pos_names.append(name)
                else: #negative
                        neg_features.append(feature)
                        neg_names.append(name)
        return pos_features, neg_features


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

features =np.load('cell_featurevectors_DL.npy')
names = np.load('cell_filenames.npy')

newfile = 'MLP_cell_results.csv' #where results will be printed
'''
f8 = open(newfile, 'wb')
fw8 = csv.writer(f8)
fw8.writerow(['Test Set','Kernel', 'deg', 'Negative Accuracy(Specificity)', 'Positive Accuracy(Sensitivity)', 'Total Accuracy'])
'''
hold_x_out = 1 #the number of images to hold out for each test (defines the training set)

#concatenate label to the end of each feature matrix

train_features, train_names, test_features, test_names = test_divide(features,names, x,hold_x_out)
        
positive_features, negative_features = label_divide(train_features,train_names)
positive_test_features, negative_test_features = label_divide(test_features,test_names)

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



'''
Winona Richey
Machine Learning 4720
2/17/2017
Implementing a multi-class, multilayer perceptron to classify Iris Data
  - have a weight vector for each class
  - predict the argmax
  
'''
import csv
import numpy as np
import random
import tensorflow as tf

def load_data(filename):
    lines = csv.reader(open(filename, "rU"))
    data = list(lines)
    ret_data = []
    for i in range(len(data)): #for each row data[i]
        ret_data.append([])# make a new row in return data
        array = data[i][0].split(',')
        for x in range(len(array) - 1): #for every element, x, in the row, data[i]
            ret_data[i].append(float(array[x])) #change the string to float
        ret_data[i].append(array[-1]) #add the class, the last column in array
    return ret_data

def splitdata(data, switchratio):
    switch = int(switchratio*len(data))
    train_set = data[:switch]
    test_set = data[switch:]
    return train_set, test_set

def class_split(dataset):
    classindex = -1 #class index in the attribute vector is -1
    Setosa_set = []
    Versicolor_set = []
    Virginica_set = []
    for feature_vector in dataset:
        if feature_vector[-1] == 'Iris-setosa':
            Setosa_set.append(feature_vector[:classindex])
        elif feature_vector[-1] == 'Iris-versicolor':
            Versicolor_set.append(feature_vector[:classindex])
        else:
            Virginica_set.append(feature_vector[:classindex])
    return Setosa_set, Versicolor_set, Virginica_set


def label_split(data, label_index):
    '''input a matrix, data with each row representing
       an example with a label at data[row][label_index]
       also converts labels to integers
'''
    data_array= []
    label_array= []
    for row in data:
        t = row[label_index]
        if t == "Iris-setosa":
            t = [1, 0, 0]
        elif t == "Iris-versicolor":
            t = [0, 1, 0]
        else:
            t = [0, 0, 1] #virginica
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
filename = "IrisData.csv"
label_index = 4
splitRatio = .8
learning_rate = 0.001
training_epochs = 1000
batch_size = 10
display_step = 50

# Network Parameters
n_hidden_1 = 100 # 1st layer number of nodes
n_hidden_2 = 100 # 2nd layer number of nodes
n_input = 4 #  data input size (img shape: 1*5)
n_classes = 3 #  total classes

data = load_data(filename)
train_data, test_data = splitdata(data, splitRatio)
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


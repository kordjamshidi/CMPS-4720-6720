'''
Winona Richey
Machine Learning 4720
2/6/2017
Implementing a multi-class Perceptron to classify Iris Data
  - have a weight vector for each class
  - predict the argmax
  
'''
import csv
import numpy as np
import random

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

def train(data):
    '''
    parameter:
         data - array of data [features, label]
    returns: [w1,w2,w3]
         
'''
    # initialize lists of weights, randomly
    R = float(.001)# learning rate
    epochs = 1000
    w_Setosa = random.sample(range(10), 5) #w0, w1, w2, w3, w4(bias)
    w_Versicolor = random.sample(range(10), 5)
    w_Virginica = random.sample(range(10), 5)
    
    for p in range(epochs):
        for x in data:
            #target label
            #[setosa, versicolor, virginica]
            t = x[-1]
            if t == "Iris-setosa":
                t = [1, 0, 0]
            elif t == "Iris-versicolor":
                t = [0, 1, 0]
            else:
                t = [0, 0, 1] #virginica
                
            
        
            #calculate weights = w*data row
            setosa = x[0]*w_Setosa[0] + x[1]*w_Setosa[1] + x[2]*w_Setosa[2] + x[3]*w_Setosa[3] + w_Setosa[4]
            versicolor = x[0]*w_Versicolor[0] + x[1]*w_Versicolor[1] + x[2]*w_Versicolor[2] + x[3]*w_Versicolor[3] + w_Versicolor[4]
            virginica = x[0]*w_Virginica[0] + x[1]*w_Virginica[1] + x[2]*w_Virginica[2] + x[3]*w_Virginica[3] + w_Virginica[4]

            #normalize
            tot_sum = setosa + versicolor + virginica
            setosa = setosa/tot_sum
            versicolor = versicolor/tot_sum
            virginica = virginica/tot_sum
            
            #predicted output
            o = max(max(setosa, versicolor), virginica)
            error = [float(t[0])-setosa, float(t[1])-versicolor, float(t[2])-virginica]
            x[-1] = 1 #bias
            x = np.array(x)
            errorR = np.multiply(R,error)
            
    
            #weight updates;
            '''
            if correct prediction leave weights the same
            if incorrect prediction (t!=o) update weights
                 weight = weight - error*learningrate*featurevector
                 w_name = w_name - error*R * row of x up until label
            '''
            
            if o == setosa: #predicted setosa
                if t != [1,0,0]: #"Iris-setosa":
                    weightupdate = w_Setosa + errorR[0]*x
                    
            elif o == versicolor:
                if t != [0,1,0]: #"Iris-versicolor":
                    weightupdate = w_Versicolor + errorR[1]*x
            else:
                if t != [0,0,1]: # "Iris-virginica":
                    weightupdate = w_Virginica +  errorR[2]*x
    
    return [w_Setosa, w_Versicolor, w_Virginica]


def test(data):
    '''returns a list of lists
         predictions: a list of targets and outputs
         predictions[#]    = a specific data point's target and output
         predictions[#][0] = a specific data point's expected result
         predictions[#][1] = a specific data point's predicted result
         
    '''
    num_correct = 0
    for x in data:
        predictions = []

        #target label: [setosa, versicolor, virginica]
        t_label = x[-1]
        if t_label == "Iris-setosa":
            t = [1, 0, 0]
        elif t_label == "Iris-versicolor":
            t = [0, 1, 0]
        else:
            t = [0, 0, 1] #virginica

        #calculate weights
        setosa = x[0]*w_Setosa[0] + x[1]*w_Setosa[1] + x[2]*w_Setosa[2] + x[3]*w_Setosa[3] + w_Setosa[4]
        versicolor = x[0]*w_Versicolor[0] + x[1]*w_Versicolor[1] + x[2]*w_Versicolor[2] + x[3]*w_Versicolor[3] + w_Versicolor[4]
        virginica = x[0]*w_Virginica[0] + x[1]*w_Virginica[1] + x[2]*w_Virginica[2] + x[3]*w_Virginica[3] + w_Virginica[4]

        #find prediction
        o = max(max(setosa, versicolor), virginica)

        #calculate accuracy
        if o == setosa: #predicted setosa
            o = [1,0,0]
        elif o == versicolor:
            o = [0,1,0]
        else:
            o = [0,0,1]
        predictions.append([t, o])

        #add to accuracy
        if t == o:
                num_correct +=1
                print "yay!"
    return predictions, num_correct


filename = "IrisData.csv"
splitRatio = .8

data = load_data(filename)
train_data, test_data = splitdata(data, splitRatio)
[w_Setosa, w_Versicolor, w_Virginica] = train(train_data)
predictions, num_correct = test(test_data)
print num_correct

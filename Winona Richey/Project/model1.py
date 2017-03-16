# code for training and testing data samples
import glob
import os
import numpy as np
import cv2
import ntpath
import tensorflow.python.platform
import tensorflow as tf
import requests
import csv
from PIL import Image
from sklearn import svm
import random

def get_accuracy(Training_data, Training_labels, negative_test_featues, positive_test_features):
        clf.fit(Training_data, Training_labels)

        negative_accuracy=clf.predict(negative_test_features)
        
        #there should be no nonzero elements in negative accuracy (all should be predicted as 0);   ideal: sum == 0
        #number of samples incorrectly calssified
        negativeaccuracy=np.count_nonzero(np.array(negative_accuracy))
        
        #number of samples classified correctly
        negativeaccuracy = len(negative_test_features)-negativeaccuracy

        #there should be no zero elements in positive accuracy (all should be predicted as 1);      ideal: len == sum
        positive_accuracy=clf.predict(positive_test_features)

        #this is the number of samples calssified correctly
        positiveaccuracy=np.count_nonzero(np.array(positive_accuracy)) 

        final_negative_accuracy = str(float(negativeaccuracy)/float(len(negative_test_features))*100)
        final_positive_accuracy = str(float(positiveaccuracy)/float(len(positive_test_features))*100)

        #accuracy = (number correctly classified negatives + number of correctly classified positives)/ total samples
        accuracy = float(negativeaccuracy + positiveaccuracy)/(float(len(negative_test_features)+ len(positive_test_features)))
        return final_negative_accuracy, final_positive_accuracy, accuracy

switch = .6 #.8 = 80% training/20% testing
neg_DL = np.load('featurevectors_DL_neg.npy')
pos_DL = np.load('featurevectors_DL_pos.npy')
switch = int(switch*min(len(neg_DL), len(pos_DL)))

positive_features= pos_DL[:switch]
positive_test_features = pos_DL[switch:]      
#[positive_featurescat, positive_test_featurescat] = splitDataset(pos_DL,switch)
negative_features= neg_DL[:switch]
negative_test_features = neg_DL[switch:]



'''-----------------------------------------------------------------'''
newfile = 'results6040.csv' #where results will be printed
f8 = open(newfile, 'wb')
fw8 = csv.writer(f8)
fw8.writerow(['Kernel', 'deg', 'Negative Accuracy(Specificity)', 'Positive Accuracy(Sensitivity)', 'Total Accuracy'])
 
negative_labels=np.ones(len(np.array(negative_features)))*(0)
positive_labels=np.ones(len(np.array(positive_features)))*(1)
Training_data=[]
Training_data=np.concatenate((positive_features,negative_features), axis=0)
Training_labels=np.concatenate((positive_labels, negative_labels), axis=0)

kernels = ["rbf","linear", "poly","poly","poly","poly","poly"]
deg = 2
for SVM_kern in kernels:
        if SVM_kern == "poly":
                clf = svm.SVC(kernel = SVM_kern, degree = deg)
                neg, pos, accuracy = get_accuracy(Training_data, Training_labels, negative_test_features, positive_test_features)
                fw8.writerow([SVM_kern, deg, neg, pos, accuracy])
                deg +=1
        else:
                clf = svm.SVC(kernel = SVM_kern)
                neg, pos, accuracy = get_accuracy(Training_data, Training_labels, negative_test_features, positive_test_features)
                fw8.writerow([SVM_kern, '-' , neg, pos, accuracy])
        


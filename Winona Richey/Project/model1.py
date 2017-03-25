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

def test_divide(features,names, test_image_index):
        train_features = []
        train_names = []
        test_features = []
        test_names = []
        transfer_name = "transfer_" + str(test_image_index)
        for x in range(len(features)):
                feature = features[x]
                name = names[x]
                if transfer_name == name[:10]:
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

features = np.load('featurevectors_DL.npy')
names = np.load('filenames.npy')

newfile = 'results.csv' #where results will be printed
f8 = open(newfile, 'wb')
fw8 = csv.writer(f8)
fw8.writerow(['Test Set','Kernel', 'deg', 'Negative Accuracy(Specificity)', 'Positive Accuracy(Sensitivity)', 'Total Accuracy'])
 

#define test features by whole image
#       --> ex: test set = all images derived from image one
#       each whole image produces
for x in range(1,3): #currently testing for 8 images
        
        train_features, train_names, test_features, test_names = test_divide(features,names, x)
        
        positive_features, negative_features = label_divide(train_features,train_names)
        positive_test_features, negative_test_features = label_divide(test_features,test_names)

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
                        fw8.writerow(['transfer_' + str(x), SVM_kern, deg, neg, pos, accuracy])
                        deg +=1
                else:
                        clf = svm.SVC(kernel = SVM_kern)
                        neg, pos, accuracy = get_accuracy(Training_data, Training_labels, negative_test_features, positive_test_features)
                        fw8.writerow(['transfer_' + str(x), SVM_kern, '-' , neg, pos, accuracy])
        


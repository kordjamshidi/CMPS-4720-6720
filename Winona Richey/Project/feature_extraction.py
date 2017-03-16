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

splitratio = .8
##reading in images
os.chdir("./neg_imgs")
i=0
negativesamples={}

cwd = os.getcwd()
print cwd
print "Loading negative files"
for filename in glob.glob('*.jpg'):
    image = cv2.imread(filename)
    negativesamples[i]=image
    i=i+1
 
print "Loading positive files"   
os.chdir("./../pos_imgs")
i=0
positivesamples={}
for filename in glob.glob('*.jpg'):
    image = cv2.imread(filename)
    positivesamples[i]=image
    i=i+1

       
###load pre-trained network (inceptionv3 from google trained on ImageNet2012)

os.chdir("./../")
with open('./classify_image_graph_def.pb', 'rb') as graph_file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(graph_file.read())
    tf.import_graph_def(graph_def, name='')

##input Negative samples and get feature vectors
negative_features = []
for i in range(0,len(negativesamples)):
    image=negativesamples[i]
    print i
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        negative_feature = np.squeeze(sess.run(
            softmax_tensor,
            {'DecodeJpeg:0': image}
        ))
        negative_feature=np.array(negative_feature)
        negative_features.append(negative_feature)
        
##input Positive samples and get feature vectors
positive_features=[]
for i in range(0,len(positivesamples)):
    image=positivesamples[i]
    with tf.Session() as sess:
        png_data = tf.placeholder(tf.string, shape=[])
        decoded_png = tf.image.decode_png(png_data, channels=3)
        softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        positive_feature = np.squeeze(sess.run(softmax_tensor,{'DecodeJpeg:0': image}))
        positive_feature=np.array(positive_feature)
        positive_features.append(positive_feature)

#saving feature vectors to a file (to run SVM)
np.save('featurevectors_DL_neg.npy', negative_features)
np.save('featurevectors_DL_pos.npy', positive_features)

f2.close()
f1.close()




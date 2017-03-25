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

##reading in images
os.chdir("./image_folder")
i=0
samples={}
names=[]

cwd = os.getcwd()
print cwd
print "Loading all files"
for filename in glob.glob('*.jpg'):
    image = cv2.imread(filename)
    samples[i]=image
    names.append(filename)
    i=i+1

       
#---load pre-trained network (inceptionv3 from google trained on ImageNet2012)---

os.chdir("./../")
with open('./classify_image_graph_def.pb', 'rb') as graph_file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(graph_file.read())
    tf.import_graph_def(graph_def, name='')

##input samples and get feature vectors
features = []
for i in range(0,len(samples)):
    image=samples[i]
    print i
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        feature = np.squeeze(sess.run(
            softmax_tensor,
            {'DecodeJpeg:0': image}
        ))
        feature=np.array(feature)
        features.append(feature)
        

#saving feature vectors to a file (to run SVM)
np.save('featurevectors_DL.npy', features)

#saving filenames, (in same order as feature vectors) to maintain association
# between file and whole image
np.save('filenames.npy', names)

print len(features)
print len(names)
print "^^ should be the same"



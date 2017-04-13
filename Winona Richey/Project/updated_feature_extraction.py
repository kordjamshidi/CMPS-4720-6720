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
os.chdir("./cell_image_folder")
i=0
samples={}
names=[]

cwd = os.getcwd()
print cwd
print "Loading all files"
for filename in glob.glob('*.jpg'):
    print filename
    image = cv2.imread(filename)
    samples[i]=image
    names.append(filename)
    i=i+1

       
#---load pre-trained network (inceptionv3 from google trained on ImageNet2012)---

print "Training"
os.chdir("./../")
with open('./classify_image_graph_def.pb', 'rb') as graph_file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(graph_file.read())
    tf.import_graph_def(graph_def, name='')

##input samples and get feature vectors
features = []
with t.Session() as sess:
        image=samples[0]
        softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        result = sess.run(softmax_tensor, {'DecodeJpeg:0': image})
        feature = np.squeeze(result)#removed np.squeeze
        feature=np.array(feature)
        features.append(feature)
    for i in range(0,len(samples)):
        '''result = sess.run(mul_op, feed_dict={a: result})'''
        image=samples[i]
        print i
        result = sess.run(softmax_tensor, feed_dict = {'DecodeJpeg:0': image})
        feature = np.squeeze(sess.run(
            softmax_tensor,
            {'DecodeJpeg:0': image}
        ))
        feature=np.array(feature)
        features.append(feature)


#saving feature vectors to a file (to run SVM)
np.save('larger_cell_featurevectors_DL.npy', features)

#saving filenames, (in same order as feature vectors) to maintain association
# between file and whole image
np.save('larger_cell_filenames.npy', names)



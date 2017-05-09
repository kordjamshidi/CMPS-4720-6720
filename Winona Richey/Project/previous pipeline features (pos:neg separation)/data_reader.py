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

i=0
positivetests={}
for data_entry in range(int(len(positivesamples)*splitratio)):
    rand_index = random.randint(0,len(positivesamples)-1) #pick a random index
    positivesamples[rand_index] = [] #delete from positive sampels
    positivetests[i]=image #add to test samples
    i=i+1

i=0
negativetests={}
for data_entry in range(int(len(negativesamples)*splitratio)):
    rand_index = random.randint(0,len(negativesamples)-1) #pick a random index
    negativesamples[rand_index] = [] #delete from positive sampels
    negativetests[i]=image #add to test samples
    i=i+1

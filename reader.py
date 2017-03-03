# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 21:08:38 2017

@author: wangz
"""

import numpy as np

data = np.genfromtxt('datasethousing.csv', delimiter=',')
CRIM = data[:,0]
ZN = data[:,1]
INDUS = data[:,2]
CH = data[:,3]
NOX = data[:,4]
RM = data[:,5]
AGE = data[:,6]
DIS = data[:,7]
RAD = data[:,8]
TAX = data[:,9]
PT = data[:,10]
B = data[:,11]
LASAT = data[:,12]
MEDV = data[:,13]

print(data)
print(CRIM)
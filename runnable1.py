# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 21:08:38 2017

@author: wangz
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plot

from sklearn.datasets import load_boston
boston = load_boston()

##basic feature
boston.keys()
#data description
print boston.DESCR
boston.data.shape
boston.feature_names
#housing price
boston.target


'''data = np.genfromtxt('datasethousing.csv', delimiter=',')
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

price = MEDV
features = price.shape'''

##put data in a panda data frame
bosdata = pd.DataFrame(boston.data)
bosdata.head()
#put name in head
bosdata.columns = boston.feature_names
bosdata.head()

bosdata['price'] = boston.target


'''from sklearn.metrics import r2_score

def perform(y_true, y_predict):
    
    score = r2_score(y_true, y_predict)

    return score

score = perform([4, -0.2, 3, 6, 5.7], [2.2, 3, 0, 8, 5.2])
'''

from sklearn.linear_model import LinearRegression
LR = LinearRegression()

#define X
X = bosdata.drop('price', axis = 1)
#plot between X and price
plot.scatter(bosdata.NOX, bosdata.price)
plot.scatter(bosdata.RM, bosdata.price)


LR.fit(X,bosdata.price)

LR.intercept_
LR.coef_

LR.score(X,bosdata.price)

LR.predict(X)[:20]

plot.scatter(LR.predict(X), bosdata.price)


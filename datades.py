# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 17:34:50 2017

@author: wangz
"""

import pandas as pd
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#from sklearn.svm import SVC, LinearSVC
#from sklearn.linear_model import LinearRegression

trainD = pd.read_csv('D:/machinelearning/CMPS-4720-6720/final program/train.csv')

trainD.columns

trainD.info()
print('_'*40)

print("Before", trainD.shape)

#checking for NAs
total = trainD.isnull().sum().sort_values(ascending=False)
percent = (trainD.isnull().sum()/trainD.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#delete variances with most NA
#laso delete Id for its useless
trainD = trainD.drop(['Id','Alley', 'PoolQC','Fence','MiscFeature','FireplaceQu','Utilities', 'Street','LandSlope','LotFrontage'], axis=1)
trainD.head()

#also delete the variables which is unmeasurable like general shape of property
trainD = trainD.drop(['BldgType','LotShape','LotConfig', 'Neighborhood','PavedDrive','Heating','GarageQual','GarageCond', 'Condition1','Condition2','LandContour','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','Functional','GarageType','GarageYrBlt','SaleType'], axis=1)

trainD['HeatingQC'] = trainD['HeatingQC'].map({None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}).astype(int)
trainD['ExterQual'] = trainD['ExterQual'].map({None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}).astype(int)   
trainD['KitchenQual'] = trainD['KitchenQual'].map({None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}).astype(int)
trainD['GarageFinish'] = trainD['GarageFinish'].map({None: 0,'Fin': 1, 'RFn': 2, 'Unf': 3}).astype(int)
trainD['CentralAir'] = trainD['CentralAir'].map( {'Y': 0, 'N': 1} ).astype(int)
trainD['SaleCondition'] = trainD['SaleCondition'].map( {'NA':0, 'Partial': 1, 'Normal': 2, 'Alloca': 3,'Family':4,'Abnorml':5,'AdjLand':6} ).astype(int)
trainD['HouseStyle'] = trainD['HouseStyle'].map( {'2.5Fin': 0, '2Story': 1, '1Story': 2,'SLvl':3,'2.5Unf':4,'1.5Fin':5,'SFoyer':6,'1.5Unf':7} ).astype(int)



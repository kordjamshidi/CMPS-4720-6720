import numpy as np
import reader
# Import of support vector classification from scikit-learn
from sklearn import svm

### load the data
cards = reader.cards
hand = reader.hand

### train the data
clf = svm.SVC(gamma=0.01, C=100)
x,y = cards, hand
clf.fit(x,y)

### test on examples
temp1 = np.array([1,10,1,11,1,13,1,12,1,1])
temp1 = temp1.reshape(1,-1)
temp2 = np.array([1,11,1,10,1,12,3,8,1,9])
temp2 = temp2.reshape(1,-1)
print("Prediction:",clf.predict(temp1),clf.predict(temp2))


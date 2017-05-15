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
train1 = np.array([1,10,1,11,1,13,1,12,1,1])
train1 = train1.reshape(1,-1)
train2 = np.array([1,9,1,12,1,10,1,11,1,13])
train2 = train2.reshape(1,-1)
train3 = np.array([3,13,2,7,4,11,3,11,2,11])
train3 = train3.reshape(1,-1)
test1 = np.array([1,11,1,10,1,12,3,8,1,9])
test1 = test1.reshape(1,-1)
test2 = np.array([1,1,1,13,2,4,2,3,1,12])
test2 = test2.reshape(1,-1)
test3 = np.array([4,11,3,2,1,11,1,4,4,2])
test3 = test3.reshape(1,-1)
print("Prediction on training instance 1:",clf.predict(train1)==np.array([9]))
print("Prediction on testing instance 1:", clf.predict(test1)==np.array([4]))
print("Prediction on training instance 2:",clf.predict(train2)==np.array([8]))
print("Prediction on testing instance 2:", clf.predict(test2)==np.array([0]))
print("Prediction on training instance 3:",clf.predict(train3)==np.array([3]))
print("Prediction on testing instance 3:", clf.predict(test3)==np.array([2]))
print("Accuracy is low on testing set. Need better learning algorithm.")



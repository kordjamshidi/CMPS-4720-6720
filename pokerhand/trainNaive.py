import reader
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

cards = reader.cards
hand = reader.hand

LEARNING_RATE = 0.01

scaler = StandardScaler()
clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(20,20),alpha=1e-5,learning_rate='constant',learning_rate_init=LEARNING_RATE,random_state=1)
X,y = cards, hand
clf.fit(X,y)

train1 = np.array([1,10,1,11,1,13,1,12,1,1])
train1 = train1.reshape(1,-1)
train2 = np.array([1,9,1,12,1,10,1,11,1,13])
train2 = train2.reshape(1,-1)
train3 = np.array([3,13,2,7,4,11,3,11,2,11])
train3 = train3.reshape(1,-1)
test1 = np.array([1,2,2,2,3,3,3,11,2,3])
test1 = test1.reshape(1,-1)
test2 = np.array([2,11,3,3,2,2,4,10,1,11])
test2 = test2.reshape(1,-1)
test3 = np.array([4,11,3,2,1,11,1,4,4,2])
test3 = test3.reshape(1,-1)
scaler.fit(train1)  
train1 = scaler.transform(train1)
test1 = scaler.transform(test1)
print("Prediction on training instance 1:",clf.predict(train1)==np.array([9]))
print("Prediction on testing instance 1:", clf.predict(test1)==np.array([4]))
print("Prediction on training instance 2:",clf.predict(train2)==np.array([8]))
print("Prediction on testing instance 2:", clf.predict(test2)==np.array([0]))
print("Prediction on training instance 3:",clf.predict(train3)==np.array([3]))
print("Prediction on testing instance 3:", clf.predict(test3)==np.array([2]))
print("Accuracy is low on testing set. Need better learning algorithm.")

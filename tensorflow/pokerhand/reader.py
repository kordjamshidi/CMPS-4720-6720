import numpy as np

### extract 10 predictable features 
cards = np.genfromtxt('train.txt',delimiter=",",usecols=(0,1,2,3,4,5,6,7,8,9),dtype=int)

### extract corresponding answers to the features
hand = np.genfromtxt('train.txt', delimiter=",",usecols=(10),dtype=int)

#print(cards)
#print(hand)


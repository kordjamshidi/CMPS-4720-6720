import reader
import numpy as np
import csv

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

trainfile="train.csv"
testfile="test.csv"

LEARNING_RATE = 0.1

scaler = StandardScaler()
#vector used for input transformation
c=[]
t=[]
fout=[]
l=[1,0,0,0,0,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,1,0,0,0,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,1,0,0,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,1,0,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,1,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,1,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,1,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,1,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,1,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,0,1,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,0,0,1,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,0,0,0,1,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,0,0,0,0,1]
c.append(l)
l=[1,0,0,0]
t.append(l)
l=[0,1,0,0]
t.append(l)
l=[0,0,1,0]
t.append(l)
l=[0,0,0,1]
t.append(l)
l=[1,0,0,0,0,0,0,0,0,0]
fout.append(l)
l=[0,1,0,0,0,0,0,0,0,0]
fout.append(l)
l=[0,0,1,0,0,0,0,0,0,0]
fout.append(l)
l=[0,0,0,1,0,0,0,0,0,0]
fout.append(l)
l=[0,0,0,0,1,0,0,0,0,0]
fout.append(l)
l=[0,0,0,0,0,1,0,0,0,0]
fout.append(l)
l=[0,0,0,0,0,0,1,0,0,0]
fout.append(l)
l=[0,0,0,0,0,0,0,1,0,0]
fout.append(l)
l=[0,0,0,0,0,0,0,0,1,0]
fout.append(l)
l=[0,0,0,0,0,0,0,0,0,1]
fout.append(l)


#reads the file and returns the data in raw form
def filereader(file,tt):
    with open(file,'rb') as f:
        reader = csv.reader(f)
        data_list = list(reader)
        #might need to pop the header
        data_list.pop(0)
        for i in range(len(data_list)):
            if(tt):
                data_list[i].pop(0)
            
            for j in range(len(data_list[i])):
                data_list[i][j]=int(data_list[i][j])
    return data_list

def sanitize(data_list):
    outputs=list()
    for i in range(len(data_list)):
        outputs.append((data_list[i][-1]))
        data_list[i].pop()
    return data_list,outputs

#converts inputs and outputs
def convert_input(x):
    ans=[]
    #print x,"printing x"
    ans.extend(t[x[0]-1])
    ans.extend(c[x[1]-1])
    ans.extend(t[x[2]-1])
    ans.extend(c[x[3]-1])
    ans.extend(t[x[4]-1])
    ans.extend(c[x[5]-1])
    ans.extend(t[x[6]-1])
    ans.extend(c[x[7]-1])
    ans.extend(t[x[8]-1])
    ans.extend(c[x[9]-1])
    return ans
def convert_inputs(input):
    for i in range(len(input)):
        input[i]=convert_input(input[i])
    return input
def convert_outputs(output):
    for i in range(len(output)):
        output[i]=fout[output[i]]
    return output


#reads the file and returns the data in raw form
def filereader(file,tt):
    with open(file,'rb') as f:
        reader = csv.reader(f)
        data_list = list(reader)
        #might need to pop the header
        data_list.pop(0)
        for i in range(len(data_list)):
            if(tt):
                data_list[i].pop(0)
            
            for j in range(len(data_list[i])):
                data_list[i][j]=int(data_list[i][j])
    return data_list

train_data=filereader('train.txt',False)
X,y=sanitize(train_data)

#training on atmost 20000 examples is enough

X=X[0:20000]
y=y[0:20000]
X=convert_inputs(X)
y=convert_outputs(y)
scaler = StandardScaler()
clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(20,20),alpha=1e-5,learning_rate='constant',learning_rate_init=LEARNING_RATE,random_state=1)
clf.fit(X,y)

train1 = np.array([1,10,1,11,1,13,1,12,1,1,9])
train1=convert_inputs(train1)
#train1 = train1.reshape(1,-1)
train2 = np.array([1,9,1,12,1,10,1,11,1,13,8])
train2 =convert_inputs(train2)
#train2 = train2.reshape(1,-1)
train3 = np.array([3,13,2,7,4,11,3,11,2,11,3])
#train3 = train3.reshape(1,-1)
train3=convert_inputs(train3)
test1 = np.array([1,11,1,10,1,12,3,8,1,9,4])
#test1 = test1.reshape(1,-1)
test1=convert_inputs(test1)
test2 = np.array([2,12,1,4,4,1,4,8,3,9,0])
#test2 = test2.reshape(1,-1)
test2=convert_inputs(test2)
test3 = np.array([4,10,1,4,1,3,4,3,2,10,2])
#test3 = test3.reshape(1,-1)
test3=convert_inputs(test3)
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



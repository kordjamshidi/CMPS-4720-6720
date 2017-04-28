from sklearn import preprocessing
import numpy as np
import reader


learning_rate=0.1
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

#derivative of the sigmoid function
def derivative(x):
    return (x)*(1.0-(x))
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
c

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
#convert final output back to form given in file
def convert_back(output):
    for i in range(len(output)):
        for j in range(10):
            if(output[i][j]>.5):
                output[i][j]=1
            else:
                output[i][j]=0
    for j in range(10):
        if (output[i]==np.array(fout[j])).all():
            output[i]=[j]
            break
    return output

train_data=reader.train_data
X,y=sanitize(train_data)

#training on atmost 20000 examples is enough

X=X[0:2]
y=y[0:2]
X=convert_inputs(X)
y=convert_outputs(y)
for i in range(len(X)):
    X[i]=np.array(X[i])
    y[i]=np.array(y[i])
X=np.array(X)
y=np.array(y)

test_input=reader.test_input

#preprocess the data here
test_input = convert_inputs(test_input)
for i in range(len(test_input)):
    test_input[i]=np.array(test_input[i])

test_input=np.array(test_input)
#dimensions of the layers
dim1 = len(X[0])
dim2 = 18
dim3 = 10
dim4 = 10
np.random.seed(1)
#weight vectors
weight0 = 2*np.random.random((dim1,dim2))-1
weight1 = 2*np.random.random((dim2,dim3))-1
weight2 = 2*np.random.random((dim3,dim4))-1
#train the network
for j in range(200):
  
    for k in range(len(X)):
        layer_0 = np.array([X[k]])
        layer_1 = sigmoid(np.dot(layer_0,weight0))
        layer_2 = sigmoid(np.dot(layer_1,weight1))
        layer_3 = sigmoid(np.dot(layer_2,weight2))
  
        layer_3_error = np.array([y[k]]) - layer_3
  
        layer_3_delta = layer_3_error * derivative(layer_3)

        layer_2_error = layer_3_delta.dot(weight2.T)
        layer_2_delta = layer_2_error * derivative(layer_2)
        layer_1_error = layer_2_delta.dot(weight1.T)
        layer_1_delta = layer_1_error * derivative(layer_1)
        weight2 += learning_rate*layer_2.T.dot(layer_3_delta)
        weight1 += learning_rate*layer_1.T.dot(layer_2_delta)
        weight0 += learning_rate*layer_0.T.dot(layer_1_delta)
layer_0 = test_input
layer_1 = sigmoid(np.dot(layer_0,weight0))
layer_2 = sigmoid(np.dot(layer_1,weight1))
layer_3 = sigmoid(np.dot(layer_2,weight2))
print "id,CLASS"
layer_3=convert_back(layer_3)
for x in range(len(layer_3)):
    print ",".join([str(int(x)),str(int(layer_3[x][0]))])

import numpy as np

LEARNING_RATE = 0.5
EPOCH = 100
ITERATION = 5

def code_answer(names):
    ans = np.zeros((names.shape[0], 3), dtype="int")
    for i in range(names.shape[0]):
        if names[i] == "Iris-virginica":
            ans[i][0] = 1
        elif names[i] == "Iris-versicolor":
            ans[i][1] = 1
        elif names[i] == "Iris-setosa":
            ans[i][2] = 1
    return ans

def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))

def train(input, answer, layer, weights):
    numLayers = len(layer) + 1

    if weights is None:
        weights = {}

        # init weights
        for w in range(numLayers):
            if w == 0:
                weights[w] = np.random.random((input.shape[1], layer[w]))
            elif w == len(layer + 1):
                weights[w] = np.random.random((layer[w - 1], answer.shape[1]))
            else:
                weights[w] = np.random.random((layer[w - 1], layer[w]))
    
    for m in range(EPOCH):
        prev = {}

        # run one epoch
        for i in range(input.shape[0]):
            s = {}
            delta = {}            
            error = {}  
            out = {}
            
            # feed forward
            inn = input[i]
            for w in range(numLayers):
                net = np.dot(inn, weights[w])
                s[w] = net
                out[w] = sigmoid(s[w])
                inn = out[w]
            
            # back propagate
            for w in reversed(range(numLayers)):
                if w == len(layer + 1):
                    error[w] = (answer[i] - out[w]) * sigmoid_gradient(s[w])
                else:
                    error[w] = np.dot(error[w + 1], np.transpose(weights[w + 1])) * sigmoid_gradient(s[w])
            
                prev[w] = 0
            
            # update weights
            for w in range(numLayers):
                if w == 0:
                    delta[w] = LEARNING_RATE * np.dot(error[w][:,None], input[i][:,None].T) + prev[w]
                else:
                    delta[w] = LEARNING_RATE * np.dot(error[w][:,None], out[w - 1][:,None].T) + prev[w]
                    
                prev[w] = delta[w]
                weights[w] = weights[w] + delta[w].T
    
    return weights
            
def test(input, answer, weights):
    error = 0
    for i in range(input.shape[0]):
        inn = input[i]
        for j in range(len(weights)):
            net = np.dot(inn, weights[j])
            s = net
            out = sigmoid(s)
            inn = out
            
        outcome = np.argmax(out)    
        ans = np.argmax(answer[i])
        
        if outcome != ans:
            error += 1

    print(error)
    return error
    
            
input = np.genfromtxt('iris.data.txt', delimiter=',',usecols=(0,1,2,3))
input = np.append(input, np.ones((input.shape[0], 1)), axis=1)
names = np.genfromtxt('iris.data.txt', delimiter=',',dtype='str',usecols=(4))
answer = code_answer(names)

def predict(layer, input):
    weights = None
    sum = 0
    for i in range(ITERATION):
        input = input.astype(np.float32)
        weights = train(input, answer, layer, weights)
        error = test(input, answer, weights)
        sum += error
    return str(sum/5.0)

layer = np.array([4])
print("Without hidden layer, Average misclassification: " + predict(layer, input) + "\n")

layer = np.array([4,2])
print("With hidden layer, Average misclassification: " + predict(layer, input) + "\n")
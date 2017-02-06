##load data
dataset <- iris
feature <- matrix(unlist(dataset[,1:4]),ncol=4)
label <- data.frame(rep(c(1,2,3),c(50,50,50)))
colnames(label) <-'label'

##choose the training and testing group, use 10 folds
set.seed(1234)
t <- sample(1:150,15)
testing <- feature[t,]
testing_label <- data.frame(label[t,])
training <- feature[-t,]
training_label <- data.frame(label[-t,])

w0 <- rep(1,135)
training_w <- cbind(training,w0)
testing_w <- cbind(testing,rep(1,15))

##use multiclass perceptron
## set the weighted matrix

W = matrix(rnorm(15),ncol=5)

for (i in 1:1000){
  for (n in 1:135){
    input = matrix(training_w[n,1:5])
    y = W %*% input
    ind = which.max(y)
    lab = training_label[n,]
    error = lab-ind
    if (error !=0) {
      W[ind,] = W[ind,]-0.01*input;
      W[lab,]= W[lab,]+0.01*input;
    }
  }
}

## test result
output = W %*% t(testing_w)
# get the index of the max value of each column
testind = apply(output,2,which.max)
#find the number of errors
num <- sum((testind - testing_label)!=0)
#Accurracy rate 14/15

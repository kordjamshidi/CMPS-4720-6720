clear;
clc;
addpath ('D:/machinelearning/CMPS-4720-6720/final program/final');

% get the whole data after first process 
load('HousingPrice.mat');

% transform the data matrix
data = price';

% sample number = 1460;
A = rand(1,1460);
[m,n]=sort(A);

Features0 = data(2:21,:);
Features = [Features0;ones(1,1460)];
Prices = data(22,:);
Prices = Prices/1000;

% randomly divide them into two groups test&training
% training group 1360
% test group 100
train_F = Features(:,n(1:1360));
train_P = Prices(:,n(1:1360));
test_F = Features(:,n(1361:1460));
test_P = Prices(:,n(1361:1460));

%input normalizing
[trainf,fs]=mapminmax(train_F);
[trainp,ps]=mapminmax(train_P);

%mapminmax('reverse',,fs)

%the nodes is decided by function by Hornik:
%N=[(2n + m)^(1/2) ,2n+m]
%N = number of nodes in hidden layer
%n = number of nodes in input layer
%m = number of nodes in output layer
%create BP neural network using traningdx
BPnet = newff(trainf,trainp,[25],{'tansig','purelin'},'traingdx');

BPnet.trainParam.max_fail=100;
BPnet.trainParam.epochs=10000;
BPnet.trainParam.goal = 1e-8;

BPnet=train(BPnet,trainf,trainp);
%  对 BP 网络进行仿真


%  计算仿真误差 
E = trainp - BPnet(trainf);
MSE=mse(E);
% predict BP
[testf,tfs]=mapminmax(test_F);
[testp,tps]=mapminmax(test_P);
outputs=BPnet(testf);
predictP = mapminmax('reverse',outputs,tps);

error=test_P-predictP;

MSE = perform(BPnet,trainp,BPnet(trainf))

%use gradient descent 
BPnet_GD = newff(trainf,trainp,[30],{},'traingd');

BPnet_GD.trainParam.max_fail=200;
BPnet_GD.trainParam.epochs=50000;
BPnet_GD.trainParam.goal = 1e-5;

BPnet_GD=train(BPnet_GD,trainf,trainp);

E = trainp - BPnet_GD(trainf);
MSE=mse(E)

outputs=BPnet_GD(testf);
predictP = mapminmax('reverse',outputs,tps);


error=test_P-predictP;
error=mse(error);
error = error^(1/2);

%simple linear regression
linP = Prices';
linX = [Features0',ones(1460,1)];
[b,bint,r,rint,stats]= regress(linP,linX);


%compare neural network and linear regression




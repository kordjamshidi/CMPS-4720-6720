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

%，隐含层的神经元个数范围由 Hornik 提出的公式：
%N=[(2n + m)^(1/2) ,2n+m]确定，其中 N 为神经网络隐含层神经元个数，n 为输入层的神经元数，m 为输出层节点个数。
%有关研究证明，网络训练精度的提高，可以采用一个隐层，而增加其神经元数量的方法来实现，这比增加网络隐含层数的方法简单的多
%1、网络层数的确定
%理论上证明:具有偏差和至少一个S 型隐层加上一个线性输出层的神经网络，能够逼近
%任何有理函数。本文预测模型采用三层神经网络，即输入层—隐层—输出层结构[6]。
%2、隐层单元数的确定
%有关研究证明，网络训练精度的提高，可以采用一个隐层，而增加其神经元数量的方法
%来实现，这比增加网络隐含层数的方法简单的多[7]。
%3、初始权值的选取
%置隐层节点的初始值为均为分布在零附近的很小的随机值。置输出层节点所连的权值数
%的一般为+1，另一半为-1。网络节点的偏置（θ）统一设置为零。
%4、响应函数的选取
%由于Sigmoid 函数的可微性，且微分式简单，易于表示，同时它又有很好的非线性映射
%能力，所以多作为映射函数。本研究采用了Sigmoid 激活函数。
%1、训练算法的选择
%LM 算法适用于解决大、中规模问题，尤其在解决大规模问题时，LM 算法有着突出的
%优点：一次迭代能使误差大幅度下降。本研究采用了LM 算法。
%create BP neural network using LM
BPnet = newff(trainf,trainp,[40],{},'traingdx');

BPnet.trainParam.max_fail=15;
BPnet.trainParam.epochs=100000;
BPnet.trainParam.goal = 1e-8;
%BPnet.trainParam.

BPnet=train(BPnet,trainf,trainp);

%  对 BP 网络进行仿真

%  计算仿真误差 
E = trainp - BPnet(trainf);
MSE=mse(E)
% predict BP
[testf,tfs]=mapminmax(test_F);
[testp,tps]=mapminmax(test_P);
outputs=BPnet(testf);
predictP = mapminmax('reverse',outputs,tps);

error=test_P-predictP;
error=mse(error);
error = error^(1/2);



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




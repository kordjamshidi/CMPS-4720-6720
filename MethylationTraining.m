%% Individual Dataset Classification with Methylation
close all
clear all

fprintf('Semi-supervised learning with single dataset.\n\n');

%% Read Data & Rescaling & t-test

% read data
addpath ('/Users/Emma/Desktop/ImagingLab/Code_Data/Three_Way_data_organized');

load('methydata_new_proc.mat');

load('phenotype.mat');

% t-test
METHY = methydata_new_proc(:,t_selection(methydata_new_proc,A));

% organize labels
Y = ones(size(A));
Y(A<1) = -1;
clear A

%% Grouping training and testing

Num = length(Y);
cls = 1;
CV = 5;

Y_training = Y(1:150,1);
Num_training = length(Y_training);

IndexC1 = find(Y_training == -1);
IndexC2 = find(Y_training ==  1);
LC1 = floor(length(IndexC1)/CV);
LC2 = floor(length(IndexC2)/CV);

po01 = randperm(length(IndexC1),LC1);
po02 = randperm(length(IndexC2),LC2);

TeIndex = [IndexC1(po01);IndexC2(po02)];
TrIndex = setdiff([1:Num]',TeIndex);
TeIndex = sort(TeIndex);


arg_folder = [0.001,1,32,128,512];
c_folder = [0.1,0.5,1,5,25];

%% training using cross-validation

TeErr = zeros(5,5);

for iii = 1:5
    for jjj = 1:5

        arg = arg_folder(iii);
        c = c_folder(jjj);
        
W = kernelmatrix(METHY(1:150,:),2,arg);

d = zeros(Num_training,1);
for i = 1:Num_training
    for j = 1:Num_training
        d(i)= d(i)+W(i,j);
    end
end

D = diag(d);

L = D-W;

i = ones(Num_training,1);
I = diag(i);

y=Y_training;
y(TeIndex)=0;
    
f=(I+c*L).^(-1)*y;

TeMatrix = myConfusionMatrix(Y(TeIndex),f(TeIndex));
TeErr(iii,jjj) = (1-trace(TeMatrix)/length(TeIndex))*100;


    end
end



%% testing

[row column]=find(TeErr==min(min(TeErr)));

ii = row(1,1);
jj = column(1,1);

arg = arg_folder(ii);
c = c_folder(jj);

W = kernelmatrix(METHY,2,arg);

d = zeros(Num,1);
for i = 1:Num
    for j = 1:Num
        d(i)= d(i)+W(i,j);
    end
end

D = diag(d);

L = D-W;

i = ones(Num,1);
I = diag(i);

y=Y;
y(151:184,1)=0;
    
f=(I+c*L).^(-1)*y;

TeMatrix = myConfusionMatrix(Y(TeIndex),f(TeIndex));
TeError = (1-trace(TeMatrix)/length(TeIndex))*100;


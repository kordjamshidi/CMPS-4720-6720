%% Individual Dataset Classification with fMRI
close all
clear all

fprintf('Semi-supervised learning on with fMRI data.\n\n');

%% Read Data & Rescaling & t-test

display('Start loading data...');

load('fmridata_new_proc.mat');

load('phenotype.mat'); % binary labels as +1 and 0

% reduce dimensionality

display('Reduing input dimensionality using t-selection...');
FMRI = fmridata_new_proc(:,t_selection(fmridata_new_proc,A));

% organize labels 
Y = ones(size(A));
Y(A<1) = -1;
clear A

%% Quasi Cross Validation to validate model

display('Validating the algorithm...');
Num = length(Y);
CV = 5;

IndexC1 = find(Y == -1);
IndexC2 = find(Y ==  1);
LC1 = floor(length(IndexC1)/CV);
LC2 = floor(length(IndexC2)/CV);

TeMatrix =cell(CV,1);
TeErr = zeros(CV,1);
Precision = zeros(CV,1);
Recall = zeros(CV,1);


for cv = 1: CV
 
% random grouping
po01 = randperm(length(IndexC1),LC1);
po02 = randperm(length(IndexC2),LC2);

TeIndex = [IndexC1(po01);IndexC2(po02)];
TrIndex = setdiff([1:Num]',TeIndex);
TeIndex = sort(TeIndex);

% parameter from previous training
arg = 5;
c = 1;
     
W = kernelmatrix(FMRI,2,arg); % Gaussian kernel applied here

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
y(TeIndex)=0;
    
f=(I+c*L).^(-1)*y;

TeMatrix{cv} = myConfusionMatrix(Y(TeIndex),f(TeIndex));
TeErr(cv) = (1-trace(TeMatrix{cv})/length(TeIndex))*100;
Precision(cv) = TeMatrix{cv}(1,1)/(TeMatrix{cv}(1,1)+TeMatrix{cv}(2,1));
Recall(cv) = TeMatrix{cv}(1,1)/(TeMatrix{cv}(1,1)+TeMatrix{cv}(1,2));

end

AveErr = mean(TeErr);
AvePrecision = mean(Precision);
AveRecall = mean(Recall);
AveAcc = 100-AveErr;

AveErr

AvePrecision

AveRecall


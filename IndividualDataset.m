%% Individual Dataset Classification
close all
clear all

fprintf('Semi-supervised learning with single dataset.\n\n');

%% Read Data & Rescaling & t-test

% read data
addpath ('/Users/Emma/Desktop/ImagingLab/Code_Data/Three_Way_data_organized');

load('fmridata_new_proc.mat');
load('methydata_new_proc.mat');
load('snpdata_new_proc.mat');

load('phenotype.mat');

% rescaling
METHY=(methydata_new_proc-repmat(mean(methydata_new_proc),[184,1]))...
    ./repmat(std(methydata_new_proc,0,1),[184,1]);

FMRI = (fmridata_new_proc-repmat(mean(fmridata_new_proc),[184,1]))...
    ./repmat(std(fmridata_new_proc,0,1),[184,1]);

SNP = (snpdata_new_proc-repmat(mean(snpdata_new_proc),[184,1]))...
    ./repmat(std(snpdata_new_proc,0,1),[184,1]);

% t-test
METHY = METHY(:,t_selection(METHY,A));
FMRI  = FRMI(:,t_selection(FMRI,A));
SNP = SNP(:,t_selection(SNP,A));

% organize labels
Y = ones(size(A));
Y(A<1) = -1;
clear A

%% Grouping training and testing

Num = length(Y);

cls = 1;

CV = 5;

IndexC1 = find(Y==-1);
IndexC2 = find(Y== 1);
LC1 = floor(length(IndexC1)/CV);
LC2 = floor(length(IndexC2)/CV);

po01 = randperm(length(IndexC1),LC1);
po02 = randperm(length(IndexC2),LC2);

TeIndex = [IndexC1(po01);IndexC2(po02)];
TrIndex = setdiff([1:Num]',TeIndex);
TeIndex = sort(TeIndex);


%%

W = kernelmatrix(METHY,2,1);

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

c=1;
    
f=(I+c.*L)\y;

TeMatrix = myConfusionMatrix(Y(TeIndex),f(TeIndex));
TeErr = (1-trace(TeMatrix)/length(TeIndex))*100;







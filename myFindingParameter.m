%
% (C) Copyright 2004.-, HyunJung (Helen) Shin (2004-12-16).
%
function [beta, Z] = myFindingParameter(beta, C, C0, sigma, TrTrIndex, TeTrIndex, const,TrIndex,Y)
  
%% load original inputs
load('fmridata_new_proc.mat');
load('methydata_new_proc.mat');
load('snpdata_new_proc.mat');

%% establish sub_folders for training 148 samples

subFMRI = fmridata_new_proc(TrIndex,:);
subMETHY = methydata_new_proc(TrIndex,:);
subSNP = snpdata_new_proc(TrIndex,:);

subY = Y(TrIndex,:);

%% get Laplacian Matrix

L = cell(1,3);
L{1,1} = LaplacianMatrix(subFMRI,sigma.^2);
L{1,2} = LaplacianMatrix(subMETHY,sigma.^2);
L{1,3} = LaplacianMatrix(subSNP,sigma.^2);


%% Combined Laplacian
% L is a <1x3 cell> array. Each cell is a <nxn double> Laplacian matrix

K   = size(L,2); % 3
n   = size(L{1},1); % 148
nTrTr = length(TrTrIndex);  % 120
for jjj = 1:length(TeTrIndex)
    subY(TrIndex == TeTrIndex(jjj))=0;
end

%% class balancing
% Class balancing matrix with Different Costs
E = ones(n,1);
ClassBalance = ones(1,nTrTr);

% Find out-of-class indexes from training set

%idxC1 = find(YIndex(TrTrIndex)==-1);
%idxC1 = find(TrIndex(TrTrIndex)==-1);
for jjj = 1:length(TrTrIndex)
    actual_Tr(jjj) = find(TrIndex == TrTrIndex(jjj));
end
idxC1 = find(subY(actual_Tr)==-1);

% Find in-class indexes from training set

%idxC2 = setdiff([1:nTr]', idxC1);
idxC2 = find(subY(actual_Tr)== 1);

%%
% Proportions of out-of-class and in-class samples in the training set
p1= length(idxC1)/nTrTr;
p2= length(idxC2)/nTrTr;
p = max(p1,p2);
% Fill ClassBalance array TODO
ClassBalance(idxC1) = 1 + const*sign(p2-p1)*p;
ClassBalance(idxC2) = 1 + const*sign(p1-p2)*p;

%Fill E TODO

%E(TrTrIndex) = ClassBalance;
E(actual_Tr) = ClassBalance;

% Build matrix E from its diagonal
E = sparse(diag(E));

% Weight each element of Y using E
TrIndex = E*TrIndex;

% Define options for the optimization subroutine
options = optimset('Display',...
                   'iter',...
                   'GradObj',...
                   'on',...
                   'LargeScale',...
                   'off',...
                   'MaxFunEvals',...
                   70);

% Perform optimization (Find minimum of constrained nonlinear multivariable function)               
% @dualfune is the dual function defined in dualfune.m, with arguments (L,Y,E)
beta = fmincon(@dualfune,... % fun
               zeros(K,1),...% x0 solution initial point
               ones(1,K),... % A matrix for inequality constraints
               C,... % b right side of inequality constraints
               [],...% Aeq
               [],...% beq
               ones(K,1)*0.05*C,...% lb lower-bound for solution
               ones(K,1)*C0,...%ub upper-bound for solution
               [],... %nonlcon
               options,...
               L,... % @fun argument 
               TrIndex,...% @fun argument 
               E);% @fun argument 
           
CombinedL = E;

% For each data set (5 in total)
for k=1:K 
  % Add the weighted Laplacian matrix 
  CombinedL = CombinedL + beta(k)*L{k};
end 

% Z seems to be the score-vector f from the paper
Z=sparse(CombinedL\TrIndex);

return

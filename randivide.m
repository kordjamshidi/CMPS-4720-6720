function [IndexTe,IndexTr] = randivide(Y,n)

% Y is the label of the data
% n is the numer of folds the data is going to be divided into
% output is the index of testing fold and training fold as a row vector

% gives the index of training and testing
% make sure that each kind of data is taken the same percentage to be
% training and the 

Class = unique(Y);
N = length(Class);
index = cell(N,1);
for i = 1:N
    index{i,1}=find(strcmp(Y,Class(i))~=0);
end

IndexTe = [];
for i = 1:N
    L = length(index{i,1});
    randtemp = randperm(L,floor(L/n));
    temp = [index{i,1}(randtemp)]';
    IndexTe = [IndexTe,temp];
end
IndexTe = sort(IndexTe);
IndexTr = setdiff(1:length(Y),IndexTe);
    
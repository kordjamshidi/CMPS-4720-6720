function [Feature_index]=t_selection(X,Y)
 
% significance level: 5% (default)
 
[m,n]=size(X);
 
fprintf('Feature selection through t-test.\n');
fprintf('Size of data %d x %d.\n',m,n);
 
class = unique(Y);
N = length(class);
Group = cell(N,1);
 
for i = 1:N
    Group{i}=X(Y==class(i),:);
end
 
if N ~= 2
    fprintf('This function does not apply.\n');
    fprintf('C is a cell that groups the input data by label.\n');
end
 
Feature_index = [];
 
for i =1:n
    h=ttest2(Group{1}(:,i),Group{2}(:,i));
    if h==1
        Feature_index=[Feature_index,i];
    end
end
 
 
if N ~= 2
    fprintf('This function does not apply.\n');
    fprintf('C is a cell that groups the input data by label.\n');
else
    cnt = length(Feature_index);
    fprintf('Number of selected feature is %d.\n',cnt);
end
fprintf('T-test done.\n');

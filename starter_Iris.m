% main function
% one layer perceptron
% import the xlsx file manually, and the variables will be
% Irissetosa(150x1 cell); VarName1(150x1 double); VarName2(150x1 double);
% VarName3(150x1 double); VarName4(150x1 double)


%% forming input as matrix,output as double

X = [VarName1,VarName2,VarName3,VarName4];
clear VarName1 VarName2 VarName3 VarName4

Class = unique(Irissetosa);
N = length(Class);

fprintf('The data includes %d kinds of Iris:\n%s  %s  %s \n',...
    N,Class{1,1},Class{2,1},Class{3,1});
% N=3, that's actually prior knowledge, and based on this we label the data
% of different classes as 1,2 or 3. In more general cases, we can use a
% 'for' loop (for i=1:N) to replace

a = find(strcmp(Irissetosa,Class(1)));
b = find(strcmp(Irissetosa,Class(2)));
c = find(strcmp(Irissetosa,Class(3)));

Y = zeros(150,1);
Y(a,:) = 1;
Y(b,:) = 2;
Y(c,:) = 3;


%% training the perceptron

%randomly pick training(80%) and testing(20%)
[IndexTe,IndexTr] = randivide(Irissetosa,5);
fprintf('The data is devided in to training group and testing group.\n')
fprintf('The data is randomly divided into 5 groups, 1 as testing, remaining as training. \n');
fprintf('No validation set is formed here. \n');

TrInput = X(IndexTr,:);
TrY = Y(IndexTr,:);
TeInput = X(IndexTe,:);
TeY = Y(IndexTe,:);

%% training perceptron

temp = ones(120,1);
TrX = [TrInput temp];
clear temp

IterMax = 10000;
iter = 0;

w = ones(3,5);
R=0.3; %learning rate

while iter<IterMax
    
    iter = iter+1;
    
    w1 = repmat(w(1,:),120,1);
    TrOutput01 = sum(TrX.*w1,2);
        
    w2 = repmat(w(2,:),120,1);
    TrOutput02 = sum(TrX.*w2,2);

    w3 = repmat(w(3,:),120,1);
    TrOutput03 = sum(TrX.*w3,2);
        
    for i = 1:120
        [max_output,max_loc]=max([TrOutput01(i),TrOutput02(i),TrOutput03(i)]);
        if TrY(i)~=max_loc
            w(max_loc,:)=w(max_loc,:)-R*TrX(i,:);
            w(TrY(i),:)=w(TrY(i),:)+R*TrX(i,:); 
        end
        
    end
end
clear IterMax iter
    
 %% testing perceptron
 
 temp = ones(30,1);
 TeX = [TeInput,temp];
 clear temp
 
 Out01 = sum(TeX.*repmat(w(1,:),30,1),2);
 Out02 = sum(TeX.*repmat(w(2,:),30,1),2);
 Out03 = sum(TeX.*repmat(w(3,:),30,1),2);
 
 TeOutput = zeros(30,1);
 r=0;
 
 fprintf('Among the 30 testing samples, \n ');
 for i = 1:30
     [temp,TeOutput(i)]=max([Out01(i),Out02(i),Out03(i)]);
     if TeOutput(i)~=TeY(i)
         fprintf('No. %d sample in testing group is classified wrong \n',i);
     else
         r = r+1;
     end
 end
 fprintf('%d samples are classified correctly\n',r);
 TeErr = (30-r)/30*100;
 fprintf('Testing error is %3.2f%% \n',TeErr);
  
 
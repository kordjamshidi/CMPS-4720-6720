Data = textread('match.txt');
data = Data(:,1:4);
label = Data(:,5);
%testing data
testindex = [1:5,51:55,101:105];
Testdata = data(testindex,:);
Testlabel = label(testindex);
Testdata(:,5)=1;
%training data
data(testindex,:)=[];
label(testindex)=[];
data(:,5)=1;


%Initial random W
%Start with Initial weights
%Pick up training instances one by one
%Classify with current weights
%If correct, no change!
%If wrong: lower score of wrong answer, raise score of right answer
W = randn(3,5);
for l = 1 : 1000
for n = 1 : 135
    temp = data(n,:);
    y = W*temp';
    [maxval, ind]=max(y);
    true = label(n);
    Error = true - ind;
    if Error~=0
        W(ind,:) = W(ind,:)-0.01*temp;
        W(true,:)= W(true,:)+0.01*temp;
    end
end
end
%Compute Test Error
TestOutput = Testdata*W';
[val, d]=max(TestOutput');
TestError = d-Testlabel';
num = 0;
for i = 1 : 15
if TestError(i) == 0
    num = num + 1;
end
end
% Print Accuracy Rate
Accuracy_Rate = num/15

    

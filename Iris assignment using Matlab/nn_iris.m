%Iris classification assignment
%Zheng Wang
%CMPS-4720-6720

%read data file
[data1,data2,data3,data4,data5] = textread('datahw1.txt','%f%f%f%f%s','delimiter',',');
data0 = zeros(150,1);
data = [data1 data2 data3 data4];

%change output data to number(using 9th char)
for i =1:150
   if data5{i,1}(9)=='o'
       data0(i)=1;
   end
   if data5{i,1}(9)=='s'
       data0(i)=2;
   end
   if data5{i,1}(9)=='g'
       data0(i)=3;
   end
end

%initial data
maxn=120;
n=5;
m=3;

p=0;s=0;t=0;net=0;
i=0;j=0;k=0;l=0;

r=0.001;

x=zeros(maxn,n);
index=zeros(maxn,1);
testdex=zeros(150-maxn,1);
w=zeros(m,n);
y=zeros(maxn,1);
a=zeros(n,1);
b=zeros(n,1);
delta=zeros(n,1);

%choose training data
for i=1:maxn
    index(i)=floor((i-1)*150/maxn)+1;
end
for i=1:maxn
    x(i,n)=1;
    for j=1:n-1
        x(i,j)=data(index(i),j);
    end
end

%the left will be test data
k=0;
for i=1:maxn-1
    if index(i+1)>index(i)+1
        k=k+1;
        testdex(k)=index(i)+1;
    end
end
k=k+1;
testdex(k)=150;

%training process    
for l=1:m
    for i=1:maxn
        if data0(index(i))==l
            y(i)=1;            %change y into a 3 by 1 vector
        else
            y(i)=0;
        end
    end
    for i=1:n
        a(i)=0;
    end
    s=0;
    for i=1:maxn
       net=0;
       for j=1:n
          net=net+a(j)*x(i,j);
       end
       s=s+0.5*(y(i)-1/(1+exp(-net)))^2; %Error
    end
    err=1; %iteration error
    while err>1e-4 
        for i=1:n
            delta(i)=0;
            for j=1:maxn
                net=0;
                for k=1:n
                    net=net+a(k)*x(j,k);
                end
                p=1/(1+exp(-net));
                delta(i)=delta(i)+(p-y(j))*p*(1-p)*x(j,i); %gradiant
            end
            b(i)=a(i)-r*delta(i); %update weight
        end
        t=0;
        for i=1:maxn
            net=0;
            for j=1:n
                net=net+b(j)*x(i,j);
            end
            t=t+0.5*(y(i)-1/(1+exp(-net)))^2;
        end
        err=abs(s-t);
        s=t;
        for i=1:n
            a(i)=b(i);
        end
    end
    for i=1:5
        w(l,i)=a(i);
    end
end

%test the number of right answer
correct_ans=0;
for i=1:150-maxn
    l=testdex(i);
    maxvalue=-1;  maxk=0;
    for k=1:m
        net=0;
        for j=1:n-1
            net=net+w(k,j)*data(l,j);
        end
        net=net+w(k,n);
        p=1/(1+exp(-net));
        if p>maxvalue
            maxvalue=p;
            maxk=k;
        end
    end
    if maxk==data0(l)
        correct_ans=correct_ans+1;
    end
end

disp('the number of right answer during test')
correct_ans
disp('the correct rate')
correct_ans/(1-maxn)
disp('the weight matrix after training')
w




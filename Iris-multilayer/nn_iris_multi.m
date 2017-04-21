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

index=zeros(maxn,1);
testdex=zeros(150-maxn,1);

x=zeros(maxn,n);
y=zeros(maxn,m);

z=zeros(n,1);
w1=zeros(n,n-1);
w2=zeros(n,n-1);
u1=zeros(n,m);
u2=zeros(n,m);
z1=zeros(maxn,n);
z2=zeros(maxn,n);
z=zeros(1,n);
v1=zeros(maxn,m);    %as Y
v2=zeros(maxn,m);
v=zeros(1,n);

deltaw=zeros(n,n-1);
deltau=zeros(n,m);

%choose training data
for i=1:maxn
    index(i)=floor((i-1)*150/maxn)+1;
end
for i=1:maxn
    x(i,n)=1;
    for j=1:n-1
        x(i,j)=data(index(i),j);
    end
    y(i,1)=0;
    y(i,2)=0;
    y(i,3)=0;
    y(i,data0(index(i)))=1;
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
for i=1:n
    for j=1:n-1
        w1(i,j)=0;
    end
end
for j=1:n
    for k=1:3
        u1(j,k)=0;
    end
end
s=0;
for l=1:maxn
    z1(l,n)=1;
    for j=1:n-1
        net=0;
        for i=1:n
            net=net+w1(i,j)*x(l,i);
        end
        z1(l,j)=1/(1+exp(-net));
    end
    for k=1:m
        net=0;
        for j=1:n
            net=net+u1(j,k)*z(j);
        end
        v1(l,k)=1/(1+exp(-net));
        s=s+0.5*(v1(l,k)-y(l,k))^2;
    end
end

err=1;
while err>1e-5
    for i=1:n
        for j=1:n-1
            deltaw(i,j)=0;
            for l=1:maxn
                for k=1:m
                    deltaw(i,j)=deltaw(i,j)+(v1(l,k)-y(l,k))*v1(l,k)*(1-v1(l,k))*u1(j,k)*z1(l,j)*(1-z1(l,j))*x(l,i);
                end
            end
            w2(i,j)=w1(i,j)-r*deltaw(i,j);
        end
    end            
    for j=1:n
        for k=1:m
            deltau(j,k)=0;
            for l=1:maxn
                deltau(j,k)=deltau(j,k)+(v1(l,k)-y(l,k))*v1(l,k)*(1-v1(l,k))*z1(l,j);
            end
            u2(j,k)=u1(j,k)-r*deltau(j,k);
        end
    end
    t=0;
    for l=1:maxn
        z2(l,n)=1;
        for j=1:n-1
            net=0;
            for i=1:n
                net=net+w2(i,j)*x(l,i);
            end
            z2(l,j)=1/(1+exp(-net));
        end
        for k=1:3
            net=0;
            for j=1:n
                net=net+u2(j,k)*z2(l,j);
            end
            v2(l,k)=1/(1+exp(-net));
            t=t+0.5*(v2(l,k)-y(l,k))^2;
        end
    end
    err=abs(s-t);
    s=t;
    u1=u2;
    w1=w2;
    z1=z2;
    v1=v2;
end
        
    
    
    

%test the number of right answer
correct_ans=0;
for l=1:150-maxn
    for j=1:n-1
        net=w1(n,j);
        for i=1:4
            net=net+w1(i,j)*data(testdex(l),i);
        end
        z(1,j)=1/(1+exp(-net));
    end
    z(1,n)=1;
    for k=1:m
        net=0;
        for j=1:n
            net=net+u1(j,k)*z(1,j);
        end
        v(1,j)=1/(1+exp(-net));
    end
    max=-1;p=0;
    for j=1:m
        if v(1,j)>max
            max=v(1,j);
            p=j;
        end
    end
    if p==y(testdex(l))
        correct_ans=correct_ans+1;
    end
end

disp('the number of right answer during test')
correct_ans
disp('the correct rate')
correct_ans/(150-maxn)
disp('the weight matrix after training')

w1
u1





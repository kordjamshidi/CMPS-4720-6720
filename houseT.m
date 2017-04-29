addpath ('D:/machinelearning/CMPS-4720-6720/final program');

load('housePD.mat');

datax=zeros(1460,20);
datay=zeros(1460,1);

datax = AllHousCon;
datay = MSZoning;
%read datax and datay

maxn=20;maxm=5;
n=1314;
testn=146
r=1e-2;

i=0;j=0;k=0;l=0;
s=0;p=0;q=0;t=0;
err=0;

x=zeros(1314,maxn+1);
y=zeros(1314,maxm);
w=zeros(maxm,maxn+1);
delta=zeros(maxm,maxn+1);

testx=zeros(1,maxn+1);


for i=1:n
  x(i,maxn+1)=1;
  for j=1:maxn
      x(i,j)=datax(i,j);
  end
  y(i,datay(i,1))=1;
end
s=0;
for k=1:n
    for l=1:maxm
        p=0;
        for j=1:maxn+1
            p=p+w(l,j)*x(k,j);
        end
        p=1/(1+exp(-p));
        s=s+(p-y(k,l))^2;
        for j=1:maxn+1
            delta(l,j)=delta(l,j)+2*(p-y(k,l))*p*(1-p)*x(k,j);
        end
    end
end

err=1;
while err>1e-5
    for l=1:maxm
        for j=1:maxn+1
            w(l,j)=w(l,j)-r*delta(l,j);
        end
    end
    t=0;
    for k=1:n
        for l=1:maxm
            p=0;
            for j=1:maxn+1
                p=p+w(l,j)*x(k,j);
            end
            p=1/(1+exp(-p));
            t=t+(p-y(k,l))^2;
            for j=1:maxn+1
                delta(l,j)=delta(l,j)+2*(p-y(k,l))*p*(1-p)*x(k,j);
            end
        end
    end
    err=abs(t-s);
    s=t;
end


%test part

max=0;mm=0;ans=0;

for i=1:testn
    for j=1:maxn
        testx(1,j)=datax(n+i,j); 
    end
    testx(1,maxn+1)=1;
    max=-1;mm=0;
    for l=1:maxm
        p=0;
        for j=1:maxn+1
            p=p+w(l,j)*testx(1,j);
        end
        p=1/(1+exp(-p));
        if (p>max)
            max=p;
            mm=l;
        end
    end
    if (l==datay(n+i,1))
        ans=ans+1;
    end
end

%
ans
%answer = 27. it doesn't have a real relationship like in Biology.
    
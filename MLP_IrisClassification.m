clear all;
close all;
clc


%% structure

n_input=4;
n_op=3;
n_hid=4;
w_ih=rand(n_input,n_hid);
w_bhid=rand(n_hid,1);
w_ho=rand(n_hid,n_op);
learning=1;
n_pattern=3;

%% data
load fisheriris

TrX=[];

for i=1:40
    
     temp=[meas(i,:);meas(50+i,:);meas(100+i,:)]';
     
     TrX=[TrX temp];

end

a = [0.9 0.1 0.1]';
b = [0.1 0.9 0.1]';
c = [0.1 0.1 0.9]';
    
temp = [repmat(a,1,1) repmat(b,1,1) repmat(c,1,1)];

TrY=repmat(temp,[1 40]);

TeX = [meas(41:50,:);meas(91:100,:);meas(141:150,:)]';

TeY = [repmat(a,[1 10]),repmat(b,[1 10]),repmat(c,[1 10])];


iter_max=10000;

error=zeros(n_op,iter_max);

fprintf('Start training...\n')
for iter=1:iter_max
    
    for j=1:size(TrX,2)
        
        %estimated output
        
        op_w=TrX(:,j)'*w_ih;
        
        op_sig=1./(1+exp(-(op_w+w_bhid')));
        
        out=1./(1+exp(-(op_sig*w_ho)));
        
        e=TrY(:,j)'-out;
        
        delta=(out.*(1-out)).*e;
       
        %hidden layer weights updation
        
        w_ho=w_ho+learning*op_sig'*delta;
              
        delta_hid=op_sig'.*(1-op_sig)'.*(w_ho*delta');
        
        %input layer weight updations
        
        w_ih=w_ih+learning*(TrX(:,j)*delta_hid');  
        
        w_bhid=w_bhid+2*delta_hid;
        
    end
    
    
    error(:,iter)=e;
    
    if (mod(iter,200)==0)
        fprintf('number of iteration = %d\n',iter);
    end
    
    
end



sse=sum((error(:,1:iter).^2),1);

plot(sse);
title('error square plot for xor gate training');
xlabel('no of iterations');
ylabel('error.^2');

pause(1);

%% testing

fprintf('start testing...\n');
TeOut = [];

 for i=1:size(TeX,2)
      
    op_w=TeX(:,i)'*w_ih;
    op_sig=1./(1+exp(-(op_w+w_bhid')));
    TeOut(:,i)=(1./(1+exp(-(op_sig*w_ho))))';
    
 end

 
 %% visualization 

 figure;
 for i = 1:3
     subplot(1,3,i);
     hold on
     plot(TeOut(i,:),'y','LineWidth',3);
     bar(TeY(i,:),'m');
     legend('Output score','Actual label');
     xlabel('testing sample');
     ylabel('score');
 end
 
 pause(0.5);
 
 actual_class = zeros(1,30);
 predicted_class = zeros(1,30);
 
 for i = 1:30    
     actual_class(1,i)= find(TeY(:,i)==max(TeY(:,i)));
     predicted_class(1,i)= find(TeOut(:,i)==max(TeOut(:,i)));
 end

 temp = find(actual_class-predicted_class~=0);
 
 fprintf('TESTING RESULT: \n');
 fprintf('Among 30 testing group, MLP fails to classify %d samples(s) \n including\n',length(temp));
 
 for i = 1:length(temp)
     fprintf('sample No. %d \n',temp(i));
 end
 
 
 map = jet(12);
 figure;
 hold all
 
 bar(find(actual_class==1),actual_class(find(actual_class==1)),'FaceColor',map(3,:));
 bar(find(actual_class==2),actual_class(find(actual_class==2)),'FaceColor',map(6,:));
 bar(find(actual_class==3),actual_class(find(actual_class==3)),'FaceColor',map(9,:));
 
 plot(find(predicted_class==1),predicted_class(find(predicted_class==1)),'o','Color',map(4,:),'LineWidth',4);
 plot(find(predicted_class==2),predicted_class(find(predicted_class==2)),'*','Color',map(8,:),'LineWidth',4);
 plot(find(predicted_class==3),predicted_class(find(predicted_class==3)),'-','Color',map(12,:),'LineWidth',4);
 
 
 legend('setosa','versicolor','virginica',...
     'predicted_setosa','predicted_versicolor','predicted_virginica');
 xlabel('test group sample');
 ylabel('score/label');
 
 title('testing performance');
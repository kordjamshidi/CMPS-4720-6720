% feature_vector, 


clear all;
close all;
clc

n_input=4;%input('no of inputs');

n_op=3;

n_hid=4;%input('enter the no of hidden layer nodes');

w_ih = rand(n_input,n_hid);

w_bhid=rand(n_hid,1);

w_ho=rand(n_hid,n_op);


eta=1;

n_pattern=3;


load fisheriris


%% training data


training=[];

for i=1:45
    
     temp=[meas(i,:);meas(50+i,:);meas(100+i,:)]';
     
     training=[training temp];

end

%% coding (+1/-1) of 3 classes

a = [0.1 0.1 0.9]';
b = [0.1 0.9 0.1]';
c = [0.9 0.1 0.1]';

%% define targets
    
temp = [repmat(a,1,1) repmat(b,1,1) repmat(c,1,1)];

desired_out=repmat(temp,[1 45]);



iteration=10000;


error=zeros(n_op,iteration);

%% Model training

for iter=1:iteration
    
    for j=1:size(training,2)
        
        %estimated output
        
        op_w = training(:,j)'*w_ih;
        
        op_sig=1./(1+exp(-(op_w+w_bhid')));
        
        out=1./(1+exp(-(op_sig*w_ho)));
        
        e=desired_out(:,j)'-out;
        
        delta=(out.*(1-out)).*e;
       
        %hidden layer weights updation
        
        w_ho=w_ho+eta*op_sig'*delta;
        
        
        
        
%         w_bout=w_bout+2*delta;

        delta_hid=op_sig'.*(1-op_sig)'.*(w_ho*delta');
        
        %input layer weight updations
        
        w_ih=w_ih+eta*(training(:,j)*delta_hid');  
        
        w_bhid=w_bhid+2*delta_hid;
        
    end
    
    
    error(:,iter)=e;
    
end



training_error=sum((error(:,1:iter).^2),1);

plot(training_error);
title('Training Error');
xlabel('no of iterations');
ylabel('error.^2');


%%%%%%%testing%%%%%%%%%


testing = [meas(46:50,:);meas(96:100,:);meas(146:150,:)]';

out = [];



 for i=1:size(testing,2)

    
    op_w = testing(:,i)'*w_ih;
    op_sig=1./(1+exp(-(op_w+w_bhid')));
    out(:,i)=(1./(1+exp(-(op_sig*w_ho))))';
    
 end
 

desired_Testout=repmat(temp,[1 5]);
error = desired_Testout - out;
%% show test error
testing_error = sum(error.^2)


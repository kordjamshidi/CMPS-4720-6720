% Individual Dataset Classification
% October 10th
% October 13th
% October 24th: repeated testing for Methylation
% October 26th: repeated testing for fMRI (linear & polynomial)



clear all

s = pwd;


cd('..'); cd('..');
load ('Three_Way_data_organized/phenotype.mat');
load('Three_Way_data_organized/fmridata_new_proc.mat');
cd(s);

Y = ones(size(A));
Y(A<1) = -1; % organized labels
clear A
Num = length(Y);


n = ones(Num,1);
I = diag(n);

c_folder = [0.1,0.25,0.5,1,2.5,5,10,25,50,100];
arg_folder = [0.001,0.1,0.5,1,2,4,16,32,64,128,256,512];
%arg_folder = [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.01,0.1,0.5,1,];
%TeErr = zeros(10,10);
TeErr = zeros(10,1);


%SNP = snpdata_new_proc - mean(mean(snpdata_new_proc));
%SNP = snpdata_new_proc - repmat(mean(snpdata_new_proc),Num,1);
%METHY = methydata_new_proc - mean(mean(methydata_new_proc));
%METHY = methydata_new_proc - repmat(mean(methydata_new_proc),Num,1);
FMRI = fmridata_new_proc - mean(mean(fmridata_new_proc));
FMRI = fmridata_new_proc - repmat(mean(fmridata_new_proc),184,1);

cls = 1;

CV = 5;

IndexC1 = find(Y==-1);
IndexC2 = find(Y== 1);
LC1 = floor(length(IndexC1)/CV);
LC2 = floor(length(IndexC2)/CV);


N = 20;
res = zeros(4,N);

tic

for iii = 1:N

    
    fprintf('Testing fMRI %d time\n',iii);
po01 = randperm(length(IndexC1),LC1);
po02 = randperm(length(IndexC2),LC2);

TeIndex = [IndexC1(po01);IndexC2(po02)];
TrIndex = setdiff([1:Num]',TeIndex);
TeIndex = sort(TeIndex);

y=Y;
y(TeIndex)=0;


for i = 1:10
    for j = 1:10
    arg = arg_folder(j);
    %W = kernelmatrix(SNP,1,1);  
    W = kernelmatrix(FMRI,2,arg);
    %W = kernelmatrix(FMRI,3,arg);
    %W = kernelmatrix(METHY,3,arg);
        
    d = zeros(Num,1);
    for ii = 1:Num
        for jj = 1:Num
            d(ii)= d(ii)+W(ii,jj);
        end       
    end
    D = diag(d);
    L = D-W;
    
    c = c_folder(i);
    f=(I+c.*L)\y;
    % for fmri only
    %f=f-mean(f);
    %
    TeMatrix = myConfusionMatrix(Y(TeIndex),f(TeIndex));
%    TeErr(i,j) = (1-trace(TeMatrix)/length(TeIndex))*100;
    TeErr(i,1) = (1-trace(TeMatrix)/length(TeIndex))*100;


    end
end

res(1,iii) = iii;
res(2,iii) = min(min(TeErr));
[Row,Column] = find(TeErr(:,:)==min(min(TeErr(:,:))));
%[Row,Column] = find(TeErr(:,:)==min(TeErr(:,:)));
res(3,iii) = c_folder(Row(1));
%res(4,iii) = arg_folder(Column(1));

end

%end

toc





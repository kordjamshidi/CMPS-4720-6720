function L = LaplacianMatrix(G,n)

%----calculating weighted matrix----%
w = kernelmatrix(G,2,n);

[NumOfPatient,DataNumber] = size(G);
D = zeros(NumOfPatient);
for i = 1:NumOfPatient
    for j = 1:NumOfPatient
        D(i,i) = D(i,i) + w(i,j);
    end
end
L = D - w;
L = D^(-0.5) * L * D^(-0.5);
clear DataNumber


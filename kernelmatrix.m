function K = kernelmatrix(X,kernel,arg)
% kernelmatrix - construct a kernel matrix
%
% K = kernelmatrix(X,kernel,arg)
%
% The inputs are
%   X       - an n-by-Nx matrix of n real data points (of dimension Nx)
%   kernel  - indicates which kernel function to use:
%                           * 1 : linear kernel
%                           * 2 : Gaussian kernel
%                           * 3 : polynomial kernel
%   arg     - parameter of the kernel function
%                           * linear kernel     :  k(x1,x2) = x1'*x2 (value of arg doesn't matter)
%                           * Gaussian kernel   :  k(x1,x2) = exp(-0.5*(x1 - x2)'*(x1 - x2)/arg)      --- if arg is a scalar
%                                                  k(x1,x2) = exp(-(x1 - x2)'*diag(arg)*(x1 - x2))    --- if arg is a vector
%                           * polynomial kernel :  k(x1,x2) = (1 + x1'*x2)^arg
%
% and the output is
%   K       - the corresponding Gram matrix : Kij = k(X(i,:),X(j,:))

% Gert Lanckriet, June 2002.

% number of data points
n = size(X,1);
% initialize K
K = zeros(n,n);

% build K, depending on the kernel
if kernel == 1
    K = X*X';
end
if kernel == 2
    if length(arg)==1
        K=exp(-0.5/arg*sqdist(X',X'));
    else
        K=exp(-sqwdist(X',X',arg));
    end
end
if kernel == 3
    K = (1+X*X').^arg;
end
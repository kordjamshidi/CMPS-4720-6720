function d=sqwdist(a,b,w)
% SQWDIST - computes squared Euclidean weighted distance matrix
%           computes a rectangular matrix of pairwise weighted distances
% between points in A (given in columns) and points in B

% NB: very fast implementation taken from Roland Bunschoten

aa = sum(diag(w)*a.*a,1); bb = sum(diag(w)*b.*b,1); ab = a'*diag(w)*b; 
d = abs(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

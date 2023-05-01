function [A] = removeRowCol(X,row1,row2,row3,row4,col1,col2,col3,col4)
% Remove rows and columns from a matrix

X(row1:row2,:) = [];
X(:,col1:col2) = [];
X(row3:row4,:) = [];
X(:,col3:col4) = [];
A = X;
end
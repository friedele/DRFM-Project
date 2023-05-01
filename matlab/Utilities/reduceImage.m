function A = reduceImage(img)
% Remove rows and columns from a matrix
 %img = imread(img,"png");
 A = removeRowCol(img,1,5,53,59,1,9,50,55);
end

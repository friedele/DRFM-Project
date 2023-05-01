function [pattern_phitheta,phi,theta] = helperPatternImport
% This function helperPatternImport is only in support of
% CustomPatternExample. It may be removed in a future release.

%   Copyright 2011 The MathWorks, Inc.

patternData = csvread('custompattern.csv'); % import csv
% Extract phi/theta values from custom pattern
chktheta = (patternData(:,2)==patternData(1,2));
blockLen = length(chktheta(chktheta~=0));
nCols = size(patternData,1)/blockLen;
thetaBlocks = reshape(patternData(:,2),blockLen,nCols);
phiBlocks = reshape(patternData(:,1),blockLen,nCols);

theta = thetaBlocks(1,:);
phi = phiBlocks(:,1).';

pattern_phitheta = reshape(patternData(:,3),blockLen,nCols).';

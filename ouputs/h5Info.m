filename = 'trainedmodel_50Epoch.h5';
filepath = 'C:\Users\friedele\Repos\DRFM\ouputs';
fileInput = fullfile(filepath,filename);
h5disp(fileInput)
h5read(fileInput,'/optimizer_weights/Adam/dense_13/kernel/m:0')
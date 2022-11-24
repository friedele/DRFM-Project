% path = 'C:\Users\friedele\Repos\DRFM';
% targetFile = fullfile(path,'inputs','combinedTargets.xlsx');
% numTargets = 4:16;
% prf = [25 50 100 200 400 800 1600 3200];
% for i=1:1000
%     idx = randi(length(prf));
%     generateTargets(targetFile,'combined',randi([min(numTargets) max(numTargets)],1),prf(idx));
% end
%   phasedArrayRadar('C:\Users\friedele\Repos\DRFM\inputs\rangeMaskTargets.xlsx','rangemask');
%   phasedArrayRadar('C:\Users\friedele\Repos\DRFM\inputs\dopplerMaskTargets.xlsx','dopplermask');
  phasedArrayRadar('C:\Users\friedele\Repos\DRFM\inputs\combinedTargets.xlsx','combined');
  phasedArrayRadar('C:\Users\friedele\Repos\DRFM\inputs\randomTargets.xlsx','random')



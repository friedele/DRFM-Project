path = 'C:\Users\friedele\Repos\DRFM';
targetFile = fullfile(path,'inputs','rangeTargets.xlsx');
numTargets = 4:16;
prf = [25 50 100 200 400 800 1600 3200];
for i=1:1000
    idx = randi(length(prf));
    generateTargets(targetFile,'rangemask',randi([min(numTargets) max(numTargets)],1),prf(idx));
end

  % This can use pre-generated file 
  phasedArrayRadarGit('C:\Users\friedele\Repos\DRFM\inputs\rangeMaskTargets.xlsx','rangemask');
  phasedArrayRadarGit('C:\Users\friedele\Repos\DRFM\inputs\dopplerMaskTargets.xlsx','dopplermask');
  phasedArrayRadarGit('C:\Users\friedele\Repos\DRFM\inputs\combinedTargets.xlsx','combined');
  phasedArrayRadarGit('C:\Users\friedele\Repos\DRFM\inputs\randomTargets.xlsx','random');
  phasedArrayRadar('C:\Users\friedele\Repos\DRFM\inputs\targetsRandom.xlsx','random');  % Short test run with less then 5 targets



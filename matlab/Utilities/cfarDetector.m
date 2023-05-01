function [numDets,pfa] = cfar2D(sampData,rngGrid,dopGrid,minRng,maxRng,dopVal)
% Range Doppler (2D) CFAR
%   Detailed explanation goes here

% Coherent Integration
%sampData = pulsint(sampData,"coherent");

% Size set by the code
% TrainingRegionSize = 2*obj.TrainingBandSize + GuardRegionSize;

%sampData = abs(sampData).^2;
% nPoints = size(sampData);
Ntrials = 1e5;
% CUTIdx = 12;
% cfar = phased.CFARDetector('NumTrainingCells',2,'NumGuardCells',2);
% cfar.ThresholdFactor = 'Auto';
% cfar.ProbabilityFalseAlarm = exp_pfa;
% det = cfar(x,CUTIdx);
% numDets = sum(det==1);
% pfa = sum(det)/Ntrials; % Actual pfa value

%
exp_pfa = 1e-6;
npower = db2pow(-10);  % Assume 10dB SNR ratio
resp = abs(sqrt(npower/2)*sampData).^2;
cfar2D = phased.CFARDetector2D('GuardBandSize',5,'TrainingBandSize',10,...
    'ProbabilityFalseAlarm',1e-6);
cfar2D.ThresholdFactor = 'Auto';
cfar2D.ProbabilityFalseAlarm = exp_pfa;
[~,rangeIndx] = min(abs(rngGrid-[minRng maxRng]));
[~,dopplerIndx] = min(abs(dopGrid-[-dopVal/2 dopVal/2]));
[columnInds,rowInds] = meshgrid(dopplerIndx(1):dopplerIndx(2),...
    rangeIndx(1):rangeIndx(2));
CUTIdx = [rowInds(:) columnInds(:)]';
det = cfar2D(resp,CUTIdx);
numDets = sum(det==1);
pfa = sum(numDets)/Ntrials; % Actual pfa value

% Plot
% figure
% detectionMap = zeros(size(resp));
% detectionMap(rangeIndx(1):rangeIndx(2),dopplerIndx(1):dopplerIndx(2)) = ...
%   reshape(double(det),rangeIndx(2)-rangeIndx(1)+1,dopplerIndx(2)-dopplerIndx(1)+1);
% h = imagesc(dopGrid,rngGrid,detectionMap);
% xlabel('Doppler (Hz)'); ylabel('Range (m)'); title('Range Doppler CFAR Detections');
% h.Parent.YDir = 'normal';
% ramp = linspace(1,10,nPoints(1))';
% xRamp = abs(sqrt(npower*ramp./2).*sampData(:,25)).^2;
% [det,thres] = cfar(xRamp,1:length(xRamp));
%
% % plot
% plot(1:length(xRamp),xRamp,1:length(xRamp),thres,...
%   find(det),xRamp(det),'o')
% legend('Signal','Threshold','Detections','Location','Northwest')
% xlabel('Time Index')
% ylabel('Level')
end
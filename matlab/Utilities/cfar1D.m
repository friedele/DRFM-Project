function det = cfar1D(sampData,nSamples)
% Range Doppler (2D) CFAR
%   Detailed explanation goes here

% Coherent Integration
%sampData = pulsint(sampData,"coherent");

% Size set by the code
% TrainingRegionSize = 2*obj.TrainingBandSize + GuardRegionSize;

pfa = 1e-6;
npower = db2pow(-10);  % Assume 10dB SNR ratio
resp = abs(sqrt(npower/2)*sampData).^2;
Ntrials = 1e5;
% The cells that follow are divided evenly on both sides of the Cell Under
% Test (CUT).
% NumTrainingCells: The number of even cells used for the noise level estimate.
% NumGuardCells: The number of even cells used to seperate the training cells
% from the CUT.
cfar = phased.CFARDetector('NumTrainingCells',12,'NumGuardCells',2);
cfar.Method = 'OS';
cfar.ProbabilityFalseAlarm = pfa;
cfar.NoisePowerOutputPort = true;
cfar.ThresholdOutputPort =  true;
cutIdx = 1:nSamples;
det = cfar(resp,cutIdx);
% pfa = sum(det)/Ntrials; % Actual pfa value

% Plot
% figure
% detectionMap = zeros(size(resp));
% detectionMap(rangeIndx(1):rangeIndx(2),dopplerIndx(1):dopplerIndx(2)) = ...
%   reshape(double(det),rangeIndx(2)-rangeIndx(1)+1,dopplerIndx(2)-dopplerIndx(1)+1);
% h = imagesc(dopGrid,rngGrid,detectionMap);
% xlabel('Doppler (Hz)'); ylabel('Range (m)'); title('Range Doppler CFAR Detections');
% h.Parent.YDir = 'normal';
ramp = linspace(1,10,1e4)';
% xRamp = abs(sqrt(npower*ramp./2).*sampData(:,1)).^2;
% [det,thres] = cfar(xRamp,1:length(xRamp));
% 
% % plot
% plot(1:length(xRamp),xRamp,1:length(xRamp),thres,...
%   find(det),xRamp(det),'o')
% legend('Signal','Threshold','Detections','Location','Northwest')
% xlabel('Time Index')
% ylabel('Level')
end
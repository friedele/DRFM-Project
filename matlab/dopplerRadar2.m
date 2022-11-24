%% Estimate Range and Speed of Three Targets
% To estimate the range and speed of three targets, create a range-Doppler map 
% using the |phased.RangeDopplerResponse| System object™. Then use the |phased.RangeEstimator| 
% and |phased.DopplerEstimator| System objects to estimate range and speed. The 
% transmitter and receiver are collocated isotropic antenna elements forming a 
% monostatic radar system.
% 
% The transmitted signal is a linear FM waveform with a pulse repetition interval 
% (PRI) of 7.0 μs and a duty cycle of 2%. The operating frequency is 77 GHz and 
% the sample rate is 150 MHz.

clear
path = 'C:\Users\friedele\Repos\DRFM';

tgtFile = fullfile(path,'inputs','targets.xlsx');
radarFile = fullfile(path,'inputs','radar.xlsx');

% Get the simulation inputs
[targetOpt,tgt] = getParameters(tgtFile,"targets");
[radarOpt,radar] = getParameters(radarFile,"radar");

c = physconst('LightSpeed');
fs = radar.fs;
fc = radar.fc;
duty = radar.duty;
pw = radar.pw;
pri = radar.pri;
prf = radar.prf;
wbBeamforming = radar.wb;
nPulses = radar.nPulses;
nDoppler = radar.nDoppler;

%% 
% Set up the scenario parameters. The transmitter and receiver are stationary 
% and located at the origin. The targets are 500, 530, and 750 meters from the 
% radar along the _x_-axis. The targets move along the _x_-axis at speeds of –60, 
% 20, and 40 m/s. All three targets have a nonfluctuating radar cross-section 
% (RCS) of 10 dB. Create the target and radar platforms.

Numtgts = size(tgt.pos,1);
tgtPos = zeros(3,Numtgts);
tgtPos(1,:) = tgt.pos; % Short ranges
tgtVel = zeros(3,Numtgts);
tgtVel(1,:) = tgt.vel; % km/sec
tgtrcs = db2pow(10)*tgt.rcs';
tgtmotion = phased.Platform(tgtPos,tgtVel);
target = phased.RadarTarget('PropagationSpeed',c,'OperatingFrequency',fc, ...
    'MeanRCS',tgtrcs);

% Radar position is stationary
radarpos = [0;0;0];
radarvel = [0;0;0];
radarmotion = phased.Platform(radarpos,radarvel);
%% 
% Create the transmitter and receiver antennas.

nRowElements = 25;
nColElements = 25;
% txArray = phased.URA('Size',[nRowElements nColElements],'Lattice','Triangular','Taper',0.25);
%viewArray(txArray)
%nElements = getNumElements(txArray);

freqVector  = [8 10].*1e9;
[pattern_phitheta,phi,theta] = helperPatternImport;
antenna = phased.CustomAntennaElement('FrequencyVector',freqVector, ...
                              'PatternCoordinateSystem','phi-theta',...
                              'PhiAngles',phi,...
                              'ThetaAngles',theta,...
                              'MagnitudePattern',pattern_phitheta,...
                              'PhasePattern',zeros(size(pattern_phitheta)));
fmax = freqVector(end);
lambda = c/fmax;
txArray = phased.URA('Element',antenna,'Size',10,'ElementSpacing',lambda/2, 'Lattice','Triangular');
% pattern(txArray,fmax,-1:0.01:1,0,'PropagationSpeed',c, ...
%     'CoordinateSystem','UV','Type','powerdb')
axis([-1 1 -50 0]);
rxArray = clone(txArray);
%% 
% Set up the transmitter-end signal processing. Create an upsweep linear FM 
% signal with a bandwidth of one half the sample rate. Find the length of the 
% PRI in samples and then estimate the rms bandwidth and range resolution.

bw = fs/2;
waveform = phased.LinearFMWaveform('PulseWidth', pw,'SampleRate',fs, ...
    'PRF',prf,'OutputFormat','Pulses','NumPulses',1,'SweepBandwidth',fs/2);
sig = waveform();
nSamples = length(sig);
bwrms = bandwidth(waveform)/sqrt(12);
rngrms = c/bwrms;
%% 
% Set up the transmitter and radiator System object properties. The peak output 
% power is 10 W (changed to 10kW)and the transmitter gain is 36 dB.

peakpower = radar.peakPwr;
txgain = radar.txGain;
transmitter = phased.Transmitter( ...
    'PeakPower',peakpower, ...
    'Gain',txgain, ...
    'InUseOutputPort',true);
radiator = phased.Radiator( ...
    'Sensor',txArray,...
    'PropagationSpeed',c,...
    'OperatingFrequency',fc);
%% 
% Set up the free-space channel in two-way propagation mode.

channel = phased.FreeSpace( ...
    'SampleRate',fs, ...    
    'PropagationSpeed',c, ...
    'OperatingFrequency',fc, ...
    'TwoWayPropagation',true);
%% 
% Set up the receiver-end processing. Set the receiver gain and noise figure.

collector = phased.Collector( ...
    'Sensor',rxArray, ...
    'PropagationSpeed',c, ...
    'OperatingFrequency',fc);
rxgain = radar.rxGain;
noisefig = 1;
receiver = phased.ReceiverPreamp( ...
    'SampleRate',fs, ...
    'Gain',rxgain, ...
    'NoiseFigure',noisefig);
%% 
% Loop over the pulses to create a data cube of 128 pulses. For each step of 
% the loop, move the target and propagate the signal. Then put the received signal 
% into the data cube. The data cube contains the received signal per pulse. Ordinarily, 
% a data cube has three dimensions where the last dimension corresponds to antennas 
% or beams. Because only one sensor is used, the cube has only two dimensions.
% 
% The processing steps are:
%% 
% # Move the radar and targets.
% # Transmit a waveform.
% # Propagate the waveform signal to the target.
% # Reflect the signal from the target.
% # Propagate the waveform back to the radar. Two-way propagation enables you 
% to combine the return propagation with the outbound propagation.
% # Receive the signal at the radar.
% # Load the signal into the data cube.

dt = pri;
datacube = complex(zeros(nSamples,nPulses));

%% Specify Phaseshift  or Time Delay Beamformer
% Create a phase-shift beamformer. Point the mainlobe of the beamformer in a 
% specific direction with respect to the local receiver coordinate system. This 
% direction is chosen to be one through which the target passes at some time in 
% its motion. This choice lets us demonstrate how the beamformer response changes 
% as the target passes through the mainlobe.

tgtAng = [0;0];
if (wbBeamforming)
    beamformer = phased.TimeDelayBeamformer('SensorArray',rxArray,...
        'DirectionSource','Property','Direction',tgtAng,...
        'SampleRate',fs,'PropagationSpeed',c);
else
    beamformer = phased.PhaseShiftBeamformer('SensorArray',rxArray,...
        'DirectionSource','Property','Direction',tgtAng,...
        'PropagationSpeed',c,'OperatingFrequency',fc,...
        'WeightsNormalization','Preserve power', 'WeightsOutputPort',false);
end

% Setup to add noise and window the signal
t = (0:nSamples-1)';
fsignal = 0.01;
x = sin(2*pi*fsignal*t);
tWin = taylorwin(nSamples,4,-40);
scaleFactor = 2;
%% Loop through the integrated number of pulses

for n = 1:nPulses
    [sensorpos,sensorvel] = radarmotion(dt);
    [tgtPos,tgtVel] = tgtmotion(dt);
    [tgtRng,tgtAng] = rangeangle(tgtPos,sensorpos);
    sig = waveform();
    [txsig,txstatus] = transmitter(sig);
    txsig = radiator(txsig,tgtAng);
    txsig = channel(txsig,sensorpos,tgtPos,sensorvel,tgtVel);    
    tgtsig = target(txsig);   
    rxCollect = collectPlaneWave(rxArray,tgtsig,tgtAng,fc);
    noise = 1e-3*(randn(size(x)) + 1i*randn(size(x)));
    sigWin = rxCollect.*tWin;
    noiseSig = sigWin+noise;
    rxBf = beamformer(noiseSig);
    datacube(:,n) = rxBf;
end
%% 
% Display the data cube containing signals per pulse.
% figure
% imagesc([0:(nPulses-1)]*pri*1e6,[0:(nSamples-1)]/fs*1e6,abs(datacube))
% xlabel('Slow Time {\mu}s')
% ylabel('Fast Time {\mu}s')
% axis xy
%% 
% Create and display the range-Doppler image for 128 Doppler bins. The image 
% shows range vertically and speed horizontally. Use the linear FM waveform for 
% match filtering. The image is here is the range-Doppler map.

%Create a range-Doppler response object.
rangedopresp = phased.RangeDopplerResponse('SampleRate',fs, ...
    'PropagationSpeed',c,'DopplerFFTLengthSource','Property', ...
    'DopplerFFTLength',nDoppler,'DopplerOutput','Speed', ...
    'OperatingFrequency',fc);
matchingcoeff = getMatchedFilter(waveform);
[rngdopresp,rnggrid,dopgrid] = rangedopresp(datacube,matchingcoeff);

im = imagesc(dopgrid,rnggrid,10*log10(abs(rngdopresp)));
set(gca,'XTick',[], 'YTick', [])
% xlabel('Closing Speed (m/s)')
% ylabel('Range (m)')
%colorbar
ylim([0 max(tgt.pos)+1e4])
axis xy

% Write the file image output
rdFile = fullfile(path,'images','rd');
saveas(gcf,rdFile,'png')

%% 
% Because the targets lie along the positive _x_-axis, positive velocity in 
% the global coordinate system corresponds to negative closing speed. Negative 
% velocity in the global coordinate system corresponds to positive closing speed.
% 
% 
% 
% Estimate the noise power after matched filtering. Create a constant noise 
% background image for simulation purposes.

% mfgain = matchingcoeff'*matchingcoeff;
% dopgain = nPulses;
% noisebw = fs;
% noisepower = noisepow(noisebw,receiver.NoiseFigure,receiver.ReferenceTemperature);
% noisepowerprc = mfgain*dopgain*noisepower;
% noisePwr = noisepowerprc*ones(size(rngdopresp));
%% 
% Create the range and Doppler estimator objects.

% rangeestimator = phased.RangeEstimator('NumEstimatesSource','Auto', ...
%     'VarianceOutputPort',true,'NoisePowerSource','Input port', ...
%     'RMSResolution',rngrms);
% dopestimator = phased.DopplerEstimator('VarianceOutputPort',true, ...
%     'NoisePowerSource','Input port','NumPulses',nPulses);
% %% 
% % Locate the target indices in the range-Doppler image. Instead of using a CFAR 
% % detector, for simplicity, use the known locations and speeds of the targets 
% % to obtain the corresponding index in the range-Doppler image.
% 
% detidx = NaN(2,Numtgts);
% tgtRng = rangeangle(tgtPos,radarpos);
% tgtspd = radialspeed(tgtPos,tgtVel,radarpos,radarvel);
% tgtdop = 2*speed2dop(tgtspd,c/fc);
% for m = 1:numel(tgtRng)
%     [~,iMin] = min(abs(rnggrid-tgtRng(m)));
%     detidx(1,m) = iMin;
%     [~,iMin] = min(abs(dopgrid-tgtspd(m)));
%     detidx(2,m) = iMin;
% end
%% 
% Find the noise power at the detection locations.

% ind = sub2ind(size(noise),detidx(1,:),detidx(2,:));
% %% 
% % Estimate the range and range variance at the detection locations. The estimated 
% % ranges agree with the postulated ranges.
% 
% [rngest,rngvar] = rangeestimator(rngdopresp,rnggrid,detidx,noise(ind))
% %% 
% % Estimate the speed and speed variance at the detection locations. The estimated 
% % speeds agree with the predicted speeds.
% 
% [spdest,spdvar] = dopestimator(rngdopresp,dopgrid,detidx,noise(ind))
% %% 
% % 
% %
%% Mono-Static Radar Simulator
function phasedArrayRadar(tgtFile,type)
% DESCRIPTION: This function drives our radar simulation model, which
% relies on two inputs of a target file and the target type
% INPUTS:  tgtFile:  string  (Excel spreadsheet file)
%          type:  string  (types - 'rangeMask', 'dopplerMask', 'combined',
%          'random', 'real')
% EXAMPLE: phasedArrayRadar('C:\Users\friedele\Repos\DRFM\inputs\combinedTargets.xlsx','combined');
% See Also:  getParameters.m

path = 'C:\Users\friedele\Repos\DRFM';
radarFile = fullfile(path,'inputs','radar.xlsx');

% Get the simulation inputs
[~,tgt] = getParameters(tgtFile,"targets");
[~,radar] = getParameters(radarFile,"radar");

% Get each set of targets
tgt.set = find(tgt.id==1);
numImages = length(tgt.set);

for k = 1:numImages
    pwNum = tgt.pw(tgt.set(k),1); % PW for this set of targets

    % Radar parameters
    c = physconst('LightSpeed');
    fs = radar.fs(pwNum);
    fc = radar.fc(pwNum);
    duty = radar.duty(pwNum);
    pw = radar.pw(pwNum);
    pri = radar.pri(pwNum);
    prf = radar.prf(pwNum);
    peakpower = radar.peakPwr(pwNum);
    txgain = radar.txGain(pwNum);
    rxgain = radar.rxGain(pwNum);
    wbBeamforming = radar.wb(pwNum);
    nPulses = radar.nPulses(pwNum);
    nDoppler = radar.nDoppler(pwNum);
    noiseFigure = radar.noiseFigure(pwNum);
    maxTgtRange = radar.maxTgtRng(pwNum);
    minTgtRange = radar.minTgtRng(pwNum);
    deltaRange = radar.deltaRange(pwNum);
    rcs = tgt.rcs(pwNum);
    lambda = c/fc;
    maxAmbRange = c/(2*prf);

    % Set up the target parameters.
    if(k==numImages)
        tgt.dwellPos = tgt.pos(tgt.set(k):end);
        tgt.dwellVel = tgt.vel(tgt.set(k):end);
    else
        tgt.dwellPos = tgt.pos(tgt.set(k):tgt.set(k+1)-1);
        tgt.dwellVel = tgt.vel(tgt.set(k):tgt.set(k+1)-1);
    end
    Numtgts = size(tgt.dwellPos,1);
    tgtPos = zeros(3,Numtgts);
    tgtVel = zeros(3,Numtgts);

    tgtPos(1,:) = tgt.dwellPos; % km
    initTgtPos = tgtPos(1,:);
    tgtVel(1,:) = tgt.dwellVel; % km/sec
    tgtrcs = db2pow(10)*tgt.rcs(tgt.set(k),1);
    tgtmotion = phased.Platform(tgtPos,tgtVel);
    target = phased.RadarTarget('PropagationSpeed',c,'OperatingFrequency',fc, ...
        'MeanRCS',tgtrcs);

    %% Simulating Test Signals for a Radar Receiver

    pd = 0.95;           % Probability of detection
    pfa = 1e-6;          % Probability of false alarm
    max_range = 5000;    % Maximum unambiguous range
    range_res = 50;      % Required range resolution
    

    %% Monostatic Radar System Design
    % We need to define several characteristics of the radar system such as the
    % waveform, the receiver, the transmitter, and the antenna used to radiate
    % and collect the signal.

    % Another important parameter of a pulse waveform is the pulse repetition
    % frequency (PRF). The PRF is determined by the maximum unambiguous range.

    waveform = phased.LinearFMWaveform('PulseWidth', pw,'SampleRate',fs, ...
        'PRF',prf,'OutputFormat','Pulses','NumPulses',1,'SweepBandwidth',fs/2);

    % Note that we set the sampling rate as twice the bandwidth.
    %
    % *Receiver Noise Characteristics*
    %
    % We assume that the only noise present at the receiver is the thermal
    % noise, so there is no clutter involved in this simulation. The power of
    % the thermal noise is related to the receiver bandwidth. The receiver's
    % noise bandwidth is set to be the same as the bandwidth of the waveform.
    % This is often the case in real systems. We also assume that the receiver
    % has a 20 dB gain and a 0 dB noise figure.

    noise_bw = 1/pw;

    receiver = phased.ReceiverPreamp(...
        'Gain',rxgain, ...
        'NoiseFigure',noiseFigure, ...
        'SampleRate',fs,...
        'EnableInputPort',true);

  

    snr_db = [-inf, 0, 3, 10, 13];
%    rocsnr(snr_db,'SignalType','NonfluctuatingNoncoherent');

    %%
    % The ROC curves show that to satisfy the design goals of Pfa = 1e-6 and Pd
    % = 0.9, the received signal's SNR must exceed 13 dB. This is a fairly high
    % requirement and is not very practical. To make the radar system more
    % feasible, we can use a pulse integration technique to reduce the required
    % SNR.  If we choose to integrate 10 pulses, the curve can be generated as

%     rocsnr([0 3 5],'SignalType','NonfluctuatingNoncoherent',...
%         'NumPulses',nPulses);


    snr_min = albersheim(pd, pfa, nPulses);

    %%

    fc = 10e9;
    lambda = c/fc;

    peakPower = ((4*pi)^3*noisepow(1/pw)*maxTgtRange^4*...
        db2pow(snr_min))/(db2pow(2*txgain)*rcs*lambda^2);
    %%
    % Note that the resulting power is about 5 kW, which is very reasonable. In
    % comparison, if we had not used the pulse integration technique, the
    % resulting peak power would have been 33 kW, which is huge.

    %%
    % With all this information, we can configure the transmitter.

    transmitter = phased.Transmitter(...
        'Gain',txgain,...
        'PeakPower',peakpower,...
        'InUseOutputPort',true);

    %%

    % *Radiator and Collector*


    nElements = 10;
    freqVector  = [8 10].*1e9;
    elementSpacing = (c/max(freqVector))/2;
    [pattern_phitheta,phi,theta] = helperPatternImport;
    antenna = phased.CustomAntennaElement('FrequencyVector',freqVector, ...
        'PatternCoordinateSystem','phi-theta',...
        'PhiAngles',phi,...
        'ThetaAngles',theta,...
        'MagnitudePattern',pattern_phitheta,...
        'PhasePattern',zeros(size(pattern_phitheta)));
    txArray = phased.URA('Element',antenna,'Size',nElements,'ElementSpacing',elementSpacing, 'Lattice','Triangular');
    rxArray = clone(txArray);

    radarmotion = phased.Platform(...
        'InitialPosition',[0; 0; 0],...
        'Velocity',[0; 0; 0]);

    % With the antenna and the operating frequency, we define both the radiator
    % and the collector.

    radiator = phased.Radiator(...
        'Sensor',txArray,...
        'PropagationSpeed',c,...
        'OperatingFrequency',fc);

    collector = phased.Collector(...
        'Sensor',rxArray,...
        'OperatingFrequency',fc);

    %%
    % This completes the configuration of the radar system. In the following
    % sections, we will define other entities, such as the target and the
    % environment that are needed for the simulation. We will then simulate the
    % signal return and perform range detection on the simulated signal.

    %% System Simulation
    % Targets
    Numtgts = size(tgt.dwellPos,1);
    tgtPos = zeros(3,Numtgts);
    tgtVel = zeros(3,Numtgts);

    tgtPos(1,:) = tgt.dwellPos; % km
    tgtVel(1,:) = tgt.dwellVel; % km/sec
    tgtrcs = db2pow(10)*tgt.rcs(tgt.set(k),1);
    tgtmotion = phased.Platform(tgtPos,tgtVel);
    target = phased.RadarTarget('PropagationSpeed',c,'OperatingFrequency',fc, ...
        'MeanRCS',tgtrcs);

    % To simulate the signal, we also need to define the propagation channel
    % between the radar system and each target.
    channel = phased.FreeSpace( ...
        'SampleRate',fs, ...
        'MaximumDistanceSource','Property',...
        'MaximumDistance',maxTgtRange,...
        'PropagationSpeed',c, ...
        'OperatingFrequency',fc, ...
        'TwoWayPropagation',true);
  
    %% Specify Phaseshift  or Time Delay Beamformer
    % Create a time delay or phase-shift beamformer, depending on the bandwidth.
    % Point the mainlobe of the beamformer in a
    % specific direction with respect to the local receiver coordinate system. This
    % direction is chosen to be one through which the target passes at some time in
    % its motion. This choice lets us demonstrate how the beamformer response changes
    % as the target passes through the mainlobe.

    tgtAng = [0;0];
    if (wbBeamforming)
        beamformer = phased.TimeDelayBeamformer('SensorArray',rxArray,...
            'DirectionSource','Property','Direction',tgtAng,...
            'SampleRate',fs);
    else
        beamformer = phased.PhaseShiftBeamformer('SensorArray',rxArray,...
            'DirectionSource','Property','Direction',tgtAng,...
            'OperatingFrequency',fc,...
            'WeightsNormalization','Preserve power');
    end
    %% Signal

    fasttime = unigrid(0,1/fs,1/prf,'[)');
    slowtime = (0:nPulses-1)/prf;

    % We set the seed for the noise generation in the receiver so that we can
    % reproduce the same results.

    receiver.SeedSource = 'Property';
    receiver.Seed = 2007;

    % Pre-allocate array for improved processing speed
    datacube = zeros(numel(fasttime),nPulses);

    sig = waveform();
    nSamples = length(sig);

     % Setup a noise jammer
    jam = 0;
    jammerloc = tgtPos(:,1);
    [~,jamang] = rangeangle(jammerloc);
    jammer = barrageJammer('ERP',1e9,...
    'SamplesPerFrame',waveform.NumPulses*waveform.SampleRate/waveform.PRF);
    jammerchannel = phased.FreeSpace('TwoWayPropagation',false,...
    'SampleRate',fs,'OperatingFrequency', fc);
    % Random noise jammer signal
    jamsig = jammer();
    jamsig = jammerchannel(jamsig,jammerloc,[0;0;0],[0;0;0],[0;0;0]);
    % Collect the jamming signal
    collector = phased.Collector('Sensor',rxArray,...
    'OperatingFrequency',fc);
    jamSig = collector(jamsig,jamang);

    %% Loop through the integrated number of pulses
    for n = 1:nPulses
        % Update the radar and target positions
        [sensorpos,sensorvel] = radarmotion(1/prf);
        [tgtPos,tgtVel] = tgtmotion(1/prf);
        [~,tgtAng] = rangeangle(tgtPos,sensorpos);

        % Simulate the pulse propagation towards the target
        [txsig,txstatus] = transmitter(sig);
        txsig = radiator(txsig,tgtAng);
        txsig = channel(txsig,sensorpos,tgtPos,sensorvel,tgtVel);

        % Reflect pulse off the target
        tgtsig = target(txsig);

        % Receive target returns at sensor
        rxCollect = collector(tgtsig,tgtAng);
        rxCollectJam = rxCollect+jamSig;
        if (jam)
            rxBf = beamformer(rxCollectJam);
        else
           rxBf = beamformer(rxCollect); 
        end
        datacube(:,n) = rxBf;
    end

    %% Range Detection

    npower = noisepow(noise_bw,receiver.NoiseFigure,...
        receiver.ReferenceTemperature);
    threshold = npower * db2pow(npwgnthresh(pfa,nPulses,'coherent'));


    matchingcoeff = getMatchedFilter(waveform);
    matchedfilter = phased.MatchedFilter(...
        'Coefficients',matchingcoeff,...
        'GainOutputPort',true);
    [rxpulses, mfgain] = matchedfilter(datacube);


    matchingdelay = size(matchingcoeff,1)-1;
    rxpulses = buffer(rxpulses(matchingdelay+1:end),size(rxpulses,1));

    %%
    % The threshold is then increased by the matched filter processing gain.
    threshold = threshold * db2pow(mfgain);

    %%
    % The following plot shows the same two pulses after they pass through the
    % matched filter.
%     helperRadarPulsePlot(rxpulses,threshold,...
%         fasttime,slowtime,num_pulse_plot);


    range_gates = c*fasttime/2;

    tvg = phased.TimeVaryingGain(...
        'RangeLoss',2*fspl(range_gates,lambda),...
        'ReferenceLoss',2*fspl(max_range,lambda));

    rxpulses = tvg(rxpulses);

    %%


    rxpulses = pulsint(rxpulses,'noncoherent');
    %%

    [~,range_detect] = findpeaks(rxpulses,'MinPeakHeight',sqrt(threshold));


      %% Detection Processing using typical CFAR threshold
        [det,pfa]= cfarDetector(datacube);
        fprintf('Number of detections(CFAR): %d\n',det);
        fprintf('PFA: %f\n',pfa);

        tgtspd = radialspeed(tgtPos,tgtVel,radarpos,radarvel);
        dopplershift = speed2dop(tgtVel,lambda);
        speed = dop2speed(dopplershift,lambda);
        tgtdop = speed2dop(tgtspd,c/fc);
    
        fprintf ('Max Unambigious Range (km) (x-axis): %.2f\n',maxAmbRange/1e3);
        fprintf ('Overall Range Shift (km) (x-axis): %.2f\n',abs(tgtPos(1,1)/1e3-initTgtPos(1,1)/1e3));
        fprintf ('Target Vel (m/s) (x-axis): %.2f\n',tgtspd);
        fprintf ('Doppler Shift (Hz) (x-axis): %.2f\n',tgtdop);
        fprintf ('True Target Vel (m/s) (x-axis): %.2f\n',speed(1,1));
        fprintf ('True Doppler Shift (Hz) (x-axis): %.2f\n',dopplershift(1,1));

    %%
    % Create and display the range-Doppler image for 64 Doppler bins. The image
    % shows range vertically and speed horizontally. Use the linear FM waveform for
    % match filtering. The image is here is the range-Doppler map.

    %Create a range-Doppler response object.
    % rangedopresp = phased.RangeDopplerResponse('SampleRate',fs, ...
    %     'PropagationSpeed',c,'DopplerOutput','Speed', ...
    %     'OperatingFrequency',fc,...
    %     'PRFSource','Property','PRF',prf);

    dopResponse = phased.RangeDopplerResponse('DopplerFFTLengthSource','Property', ...
        'DopplerFFTLength',4096, ...
        'SampleRate',fs,'DopplerOutput','Frequency');

    velResponse = phased.RangeDopplerResponse('DopplerFFTLengthSource','Property', ...
        'DopplerFFTLength',4096, ...
        'SampleRate',fs,'DopplerOutput','Speed',...
        'OperatingFrequency',fc);

    matchingcoeff = getMatchedFilter(waveform);
    % [rngDopresp,rngGrid,dopGrid] = rangedopresp(datacube,matchingcoeff);
     %    [rngdopresp,rnggrid,dopgrid] = velResponse(datacube,matchingcoeff); % Best response
    [Rngdopresp,Rnggrid,Dopgrid] = dopResponse(datacube,matchingcoeff); %

    % figure
    % imagesc(dopgrid,rnggrid,mag2db(abs(rngdopresp)));
    posDopplerShift = prf/2;
    negDopplerShift = -(posDopplerShift);
    %
    % xlabel('Velocity (m/s)')
    % ylabel('Range (m)')
    % colormap(jet(256));
    % colorbar
    % caxis([-30, 30]);
    % ylim([minTgtRange maxTgtRange])
    % xlim([-1 1])

    figure
    imagesc(Dopgrid/2,Rnggrid,mag2db(abs(Rngdopresp)));
    set(gca,'XTick',[], 'YTick', [])
    colormap(jet(256));
   % colorbar
    caxis([-30, 30]);
    ylim([min(tgtPos(1,:))-deltaRange max(tgtPos(1,:))+deltaRange])
    xlim([negDopplerShift/2 posDopplerShift/2])

%     xlabel('Doppler Frequency (Hz)')
%     ylabel('Range (m)')
    % figure
    % imagesc(Dopgrid,Rnggrid,mag2db(abs(Rngdopresp)));
    %imagesc(dopGrid,rngGrid,mag2db(abs(rngDopresp)));
    %set(gca,'XTick',[], 'YTick', [])
    % xlabel('Velocity (m/s)')
    % ylabel('Range (m)')
    % colorbar
    % ylim([min(tgt.pos)-1e4 max(tgt.pos)+1e4])
    % axis xy

    fprintf('Using PW%d\n',pwNum)
    tmpFile = fullfile(path,'images', 'tmpImage');
    imgNum = num2str(k);
    filetype = '.png';

    %% Detection Processing using typical CFAR threshold
%     fprintf('Number of Targets: %d\n',Numtgts);
%     det = cfar1D(datacube,nSamples);
%     numDet = sum(det==1);
%     fprintf('Number of detections(CFAR-1D): %d\n',numDet(1));
%     fprintf('PFA: %f\n',pfa);

    %% Detection Processing using typical 2D range Doppler CFAR threshold
%     [det,pfa]= cfar2D(Rngdopresp,Rnggrid,Dopgrid,minTgtRange,maxTgtRange,posDopplerShift);
%     fprintf('Number of detections(CFAR-2D): %d\n',det);
%     fprintf('PFA: %f\n',pfa);

    % Write the file image output
        % Write the file image output
    switch type
        case ('random')
            label = 'n01_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n01_randomTgts/', filename);
        case ('rangemask')
            label = 'n02_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n02_rangeTgts',filename);
        case('dopplermask')
            label = 'n03_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n03_dopplerTgts/',filename);
        case('combined')
            label = 'n04_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n04_combinedTgts/',filename);
        case ('real')
            filename = 'n05_';
            imgFile = fullfile(path,'images/n05_realTgts//',filename);
        case('noise')
            label = 'n06_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n06_noiseOnly/',filename);
        otherwise
            disp('Choices are random, rangemask, dopplermask, combined', 'real', 'noise')
            return
    end

    saveas(gcf,tmpFile,'png')
    img = im2double(imread(tmpFile,"png"));
%     grayImage = im2gray(img);
%     gray64Image = imresize(grayImage,'OutputSize',[64,64]);  % Should I use a down sampling technique? Yes
%     gray64Image = reduceImage(gray64Image);
%     imwrite(gray64Image,imgFile,"png");
    close  % Current figure
end
end
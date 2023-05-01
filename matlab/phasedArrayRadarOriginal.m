<<<<<<< HEAD
%% Phased Array Radar Simulator
function phasedArrayRadar(tgtFile,type)
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

    % Radar position is stationary
    radarpos = [0;0;0];
    radarvel = [0;0;0];
    radarmotion = phased.Platform(radarpos,radarvel);
    %%
    % Create the transmitter and receiver antennas.

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
    %%
    % Set up the transmitter-end signal processing. Create an upsweep linear FM
    % signal with a bandwidth of one half the sample rate. Find the length of the
    % PRI in samples and then estimate the rms bandwidth and range resolution.

    waveform = phased.LinearFMWaveform('PulseWidth', pw,'SampleRate',fs, ...
        'PRF',prf,'OutputFormat','Pulses','NumPulses',1,'SweepBandwidth',fs/2);
    sig = waveform();
    nSamples = length(sig);
    bwrms = bandwidth(waveform)/sqrt(12);
    rngrms = c/bwrms;
    %%
    % Set up the transmitter and radiator System object properties.

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
        'MaximumDistanceSource','Property',...
        'MaximumDistance',maxTgtRange,...
        'PropagationSpeed',c, ...
        'OperatingFrequency',fc, ...
        'TwoWayPropagation',true);
    %%
    % Set up the receiver-end processing. Set the receiver gain and noise figure.

    collector = phased.Collector( ...
        'Sensor',rxArray, ...
        'OperatingFrequency',fc);
    receiver = phased.ReceiverPreamp( ...
        'SampleRate',fs, ...
        'Gain',rxgain, ...
        'NoiseFigure',noiseFigure);
    %%
    % Loop over the pulses to create a data cube of 64 pulses. For each step of
    % the loop, move the target and propagate the signal. Then put the received signal
    % into the data cube. The data cube contains the received signal per pulse.
    %
    % The processing steps are:  Block diagram would be helpful and detail
    % in the paper
    %%
    % # Propagate the targets.
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

    % Setup to add noise and window the signal
    t = (0:nSamples-1)';
    fsignal = 0.01;
    x = sin(2*pi*fsignal*t);
    tWin = taylorwin(nSamples,4,-40);

    %% Loop through the integrated number of pulses
    for n = 1:nPulses
        [sensorpos,sensorvel] = radarmotion(dt);
        [tgtPos,tgtVel] = tgtmotion(dt);
        [~,tgtAng] = rangeangle(tgtPos,sensorpos);
        sig = waveform();
        [txsig,~] = transmitter(sig);
        txsig = radiator(txsig,tgtAng);
        txsig = channel(txsig,sensorpos,tgtPos,sensorvel,tgtVel);
        tgtsig = target(txsig);  %error
        rxCollect = collectPlaneWave(rxArray,tgtsig,tgtAng,fc);
        noise = noiseFigure*(randn(size(x)) + 1i*randn(size(x)));
        sigWin = rxCollect.*tWin;
        noiseSig = sigWin+noise;
        rxBf = beamformer(noiseSig);
        datacube(:,n) = rxBf;
    end
    %     tgtspd = radialspeed(tgtPos,tgtVel,radarpos,radarvel);
    %     dopplershift = speed2dop(tgtVel,lambda);
    %     speed = dop2speed(dopplershift,lambda);
    %     tgtdop = speed2dop(tgtspd,c/fc);
    %
    %     fprintf ('Max Unambigious Range (km) (x-axis): %.2f\n',maxAmbRange/1e3);
    %     fprintf ('Overall Range Shift (km) (x-axis): %.2f\n',abs(tgtPos(1,1)/1e3-initTgtPos(1,1)/1e3));
    %     fprintf ('Target Vel (m/s) (x-axis): %.2f\n',tgtspd);
    %     fprintf ('Doppler Shift (Hz) (x-axis): %.2f\n',tgtdop);
    %     fprintf ('True Target Vel (m/s) (x-axis): %.2f\n',speed(1,1));
    %     fprintf ('True Doppler Shift (Hz) (x-axis): %.2f\n',dopplershift(1,1));
    % Determine the range bins
    %     fasttime = unigrid(0,1/fs,1/prf,'[)');
    %     rangebins = (physconst('LightSpeed')*fasttime)/2;
    %     %
    %     probfa = 1e-9;
    %     NoiseBandwidth = 5e6/2;
    %     npower = noisepow(NoiseBandwidth,...
    %         receiver.NoiseFigure,receiver.ReferenceTemperature);
    %     thresh = npwgnthresh(probfa,nPulses,'coherent');
    %     thresh = sqrt(npower*db2pow(thresh));
    %     [~,range_detect] = findpeaks(pulsint(abs(datacube),'coherent'),...
    %         'MinPeakHeight',thresh,'SortStr','descend');
    %
    %     range_estimate = rangebins(range_detect(2));
    %     ts = datacube(range_detect(1),:).';
    %     [Pxx,F] = periodogram(ts,[],1024,prf,'centered');
    %     % % plot(F,10*log10(Pxx))
    %     %
    %     [~,I] = max(Pxx);
    %     lambda = physconst('LightSpeed')/fc;
    %     tgtspeed = dop2speed(F(I)/2,lambda);
    %     fprintf('Estimated range of the target is %4.2f km.\n',...
    %         range_estimate/1e3)
    %     fprintf('Estimated target speed is %3.1f m/sec.\n',tgtspeed)
    %%
    % Display the data cube containing signals per pulse.
    % figure
    % imagesc([0:(nPulses-1)]*pri*1e6,[0:(nSamples-1)]/fs*1e6,abs(datacube))
    % xlabel('Slow Time {\mu}s')
    % ylabel('Fast Time {\mu}s')
    % axis xy
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
    %     [rngdopresp,rnggrid,dopgrid] = velResponse(datacube,matchingcoeff); % Best response
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
    %     colormap(jet(256));
    %     colorbar
    %     caxis([-30, 30]);
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

    % Write the file image output
    switch type
        case ('random')
            label = 'n01_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n01_randomTgts/', filename);
        case ('rangemask')
            label = 'n02_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n02_rangeTgts/',filename);
        case('dopplermask')
            label = 'n03_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n03_dopTgts/',filename);
        case ('real')
            filename = 'rd';
            imgFile = fullfile(path,'images/realTgts/',filename);
        case('combined')
            label = 'n04_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n04_combinedTgts/',filename);
        otherwise
            disp('Choices are random, rangemask, dopplermask, combined', 'real')
            return
    end

    saveas(gcf,tmpFile,'png')
    img = im2double(imread(tmpFile,"png"));
    grayImage = im2gray(img);
    gray64Image = imresize(grayImage,'OutputSize',[64,64]);  % Should I use a down sampling technique?
    imwrite(gray64Image,imgFile,"png");
    close  % Current figure
end
=======
%% Phased Array Radar Simulator
function phasedArrayRadar(tgtFile,type)
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

    % Radar position is stationary
    radarpos = [0;0;0];
    radarvel = [0;0;0];
    radarmotion = phased.Platform(radarpos,radarvel);
    %%
    % Create the transmitter and receiver antennas.

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
    %%
    % Set up the transmitter-end signal processing. Create an upsweep linear FM
    % signal with a bandwidth of one half the sample rate. Find the length of the
    % PRI in samples and then estimate the rms bandwidth and range resolution.

    waveform = phased.LinearFMWaveform('PulseWidth', pw,'SampleRate',fs, ...
        'PRF',prf,'OutputFormat','Pulses','NumPulses',1,'SweepBandwidth',fs/2);
    sig = waveform();
    nSamples = length(sig);
    bwrms = bandwidth(waveform)/sqrt(12);
    rngrms = c/bwrms;
    %%
    % Set up the transmitter and radiator System object properties.

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
        'MaximumDistanceSource','Property',...
        'MaximumDistance',maxTgtRange,...
        'PropagationSpeed',c, ...
        'OperatingFrequency',fc, ...
        'TwoWayPropagation',true);
    %%
    % Set up the receiver-end processing. Set the receiver gain and noise figure.

    collector = phased.Collector( ...
        'Sensor',rxArray, ...
        'OperatingFrequency',fc);
    receiver = phased.ReceiverPreamp( ...
        'SampleRate',fs, ...
        'Gain',rxgain, ...
        'NoiseFigure',noiseFigure);
    %%
    % Loop over the pulses to create a data cube of 64 pulses. For each step of
    % the loop, move the target and propagate the signal. Then put the received signal
    % into the data cube. The data cube contains the received signal per pulse.
    %
    % The processing steps are:  Block diagram would be helpful and detail
    % in the paper
    %%
    % # Propagate the targets.
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

    % Setup to add noise and window the signal
    t = (0:nSamples-1)';
    fsignal = 0.01;
    x = sin(2*pi*fsignal*t);
    tWin = taylorwin(nSamples,4,-40);

    %% Loop through the integrated number of pulses
    for n = 1:nPulses
        [sensorpos,sensorvel] = radarmotion(dt);
        [tgtPos,tgtVel] = tgtmotion(dt);
        [~,tgtAng] = rangeangle(tgtPos,sensorpos);
        sig = waveform();
        [txsig,~] = transmitter(sig);
        txsig = radiator(txsig,tgtAng);
        txsig = channel(txsig,sensorpos,tgtPos,sensorvel,tgtVel);
        tgtsig = target(txsig);  %error
        rxCollect = collectPlaneWave(rxArray,tgtsig,tgtAng,fc);
        noise = noiseFigure*(randn(size(x)) + 1i*randn(size(x)));
        sigWin = rxCollect.*tWin;
        noiseSig = sigWin+noise;
        rxBf = beamformer(noiseSig);
        datacube(:,n) = rxBf;
    end
    %     tgtspd = radialspeed(tgtPos,tgtVel,radarpos,radarvel);
    %     dopplershift = speed2dop(tgtVel,lambda);
    %     speed = dop2speed(dopplershift,lambda);
    %     tgtdop = speed2dop(tgtspd,c/fc);
    %
    %     fprintf ('Max Unambigious Range (km) (x-axis): %.2f\n',maxAmbRange/1e3);
    %     fprintf ('Overall Range Shift (km) (x-axis): %.2f\n',abs(tgtPos(1,1)/1e3-initTgtPos(1,1)/1e3));
    %     fprintf ('Target Vel (m/s) (x-axis): %.2f\n',tgtspd);
    %     fprintf ('Doppler Shift (Hz) (x-axis): %.2f\n',tgtdop);
    %     fprintf ('True Target Vel (m/s) (x-axis): %.2f\n',speed(1,1));
    %     fprintf ('True Doppler Shift (Hz) (x-axis): %.2f\n',dopplershift(1,1));
    % Determine the range bins
    %     fasttime = unigrid(0,1/fs,1/prf,'[)');
    %     rangebins = (physconst('LightSpeed')*fasttime)/2;
    %     %
    %     probfa = 1e-9;
    %     NoiseBandwidth = 5e6/2;
    %     npower = noisepow(NoiseBandwidth,...
    %         receiver.NoiseFigure,receiver.ReferenceTemperature);
    %     thresh = npwgnthresh(probfa,nPulses,'coherent');
    %     thresh = sqrt(npower*db2pow(thresh));
    %     [~,range_detect] = findpeaks(pulsint(abs(datacube),'coherent'),...
    %         'MinPeakHeight',thresh,'SortStr','descend');
    %
    %     range_estimate = rangebins(range_detect(2));
    %     ts = datacube(range_detect(1),:).';
    %     [Pxx,F] = periodogram(ts,[],1024,prf,'centered');
    %     % % plot(F,10*log10(Pxx))
    %     %
    %     [~,I] = max(Pxx);
    %     lambda = physconst('LightSpeed')/fc;
    %     tgtspeed = dop2speed(F(I)/2,lambda);
    %     fprintf('Estimated range of the target is %4.2f km.\n',...
    %         range_estimate/1e3)
    %     fprintf('Estimated target speed is %3.1f m/sec.\n',tgtspeed)
    %%
    % Display the data cube containing signals per pulse.
    % figure
    % imagesc([0:(nPulses-1)]*pri*1e6,[0:(nSamples-1)]/fs*1e6,abs(datacube))
    % xlabel('Slow Time {\mu}s')
    % ylabel('Fast Time {\mu}s')
    % axis xy
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
    %     [rngdopresp,rnggrid,dopgrid] = velResponse(datacube,matchingcoeff); % Best response
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
    %     colormap(jet(256));
    %     colorbar
    %     caxis([-30, 30]);
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

    % Write the file image output
    switch type
        case ('random')
            label = 'n01_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n01_randomTgts/', filename);
        case ('rangemask')
            label = 'n02_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n02_rangeTgts/',filename);
        case('dopplermask')
            label = 'n03_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n03_dopTgts/',filename);
        case ('real')
            filename = 'rd';
            imgFile = fullfile(path,'images/realTgts/',filename);
        case('combined')
            label = 'n04_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n04_combinedTgts/',filename);
        otherwise
            disp('Choices are random, rangemask, dopplermask, combined', 'real')
            return
    end

    saveas(gcf,tmpFile,'png')
    img = im2double(imread(tmpFile,"png"));
    grayImage = im2gray(img);
    gray64Image = imresize(grayImage,'OutputSize',[64,64]);  % Should I use a down sampling technique?
    imwrite(gray64Image,imgFile,"png");
    close  % Current figure
end
>>>>>>> 8a98476f7c6e1b6902031883daeb0f7e8cd91cca
end
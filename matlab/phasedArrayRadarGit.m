%% Phased Array Radar Simulator
function phasedArrayRadarGit(tgtFile,type)
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
    deltaRange = tgt.deltaRange(tgt.set(k),1)*2;

    % Radar parameters
    c = physconst('LightSpeed');
    fs = radar.fs(pwNum);
    fc = radar.fc(pwNum);
    pw = radar.pw(pwNum);
    pri = radar.pri(pwNum);
    prf = radar.prf(pwNum);
    peakpower = radar.peakPwr(pwNum);
    txgain = radar.txGain(pwNum);
    wbBeamforming = radar.wb(pwNum);
    nPulses = radar.nPulses(pwNum);
    noiseFigure = radar.noiseFigure(pwNum);
    maxTgtRange = radar.maxTgtRng(pwNum);
   
    % Set up the target parameters.
    if(k==numImages)
        tgt.dwellPos = tgt.pos(tgt.set(k):end);
        tgt.dwellVel = tgt.vel(tgt.set(k):end);
    else
        tgt.dwellPos = tgt.pos(tgt.set(k):tgt.set(k+1)-1);
        tgt.dwellVel = tgt.vel(tgt.set(k):tgt.set(k+1)-1);
    end

    tgt.dwellPos(any(isnan(tgt.dwellPos),2),:)=[];
    tgt.dwellVel(any(isnan(tgt.dwellVel),2),:)=[];
    Numtgts = size(tgt.dwellPos,1);
    tgtPos = zeros(3,Numtgts);
    tgtVel = zeros(3,Numtgts);

    tgtPos(1,:) = tgt.dwellPos; % km
    tgtVel(1,:) = tgt.dwellVel; % km/sec
    tgtrcs = db2pow(10)*tgt.rcs(tgt.set(k),1);
    tgtmotion = phased.Platform(tgtPos,tgtVel);
    target = phased.RadarTarget('PropagationSpeed',c,'OperatingFrequency',fc, ...
        'MeanRCS',tgtrcs);

    % Radar position is stationary
    radarpos = [0;0;0];
    radarvel = [0;0;0];
    radarmotion = phased.Platform(radarpos,radarvel);

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

    dopResponse = phased.RangeDopplerResponse('DopplerFFTLengthSource','Property', ...
        'DopplerFFTLength',4096, ...
        'SampleRate',fs,'DopplerOutput','Frequency');

    velResponse = phased.RangeDopplerResponse('DopplerFFTLengthSource','Property', ...
        'DopplerFFTLength',4096, ...
        'SampleRate',fs,'DopplerOutput','Speed',...
        'OperatingFrequency',fc);

    matchingcoeff = getMatchedFilter(waveform);
    %     [rngdopresp,rnggrid,dopgrid] = velResponse(datacube,matchingcoeff); % Best response
    [Rngdopresp,Rnggrid,Dopgrid] = dopResponse(datacube,matchingcoeff); %
    posDopplerShift = prf/2;
    negDopplerShift = -(posDopplerShift);
    
    imagesc(Dopgrid/2,Rnggrid,mag2db(abs(Rngdopresp)));  
    ylim([min(tgtPos(1,:))-deltaRange max(tgtPos(1,:))+deltaRange]);
    xlim([negDopplerShift/2 posDopplerShift/2]);
    set(gca,'XTick',[], 'YTick', [])
%     xlabel('Doppler Frequency (Hz)')
%     ylabel('Range (m)')

    fprintf('Using PW%d\n',pwNum);
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
            imgFile = fullfile(path,'images/n02_rangeTgts',filename);
        case('dopplermask')
            label = 'n03_';
            filename = [label imgNum filetype];
            imgFile = fullfile(path,'images/n03_dopplerTgts/',filename);
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
   % gray64Image = reduceImage(gray64Image);
    imwrite(gray64Image,imgFile,"png");
    close  % Current figure
end
end
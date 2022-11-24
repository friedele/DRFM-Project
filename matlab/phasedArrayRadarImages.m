%% Phased Array Radar Simulator
function phasedArrayRadarImages(tgtFile)
path = 'C:\Users\friedele\Repos\DRFM';
radarFile = fullfile(path,'inputs','radar.xlsx');

% Get the simulation inputs
[~,tgt] = getParameters(tgtFile,"targets");
[~,radar] = getParameters(radarFile,"radar");

% Get each set of targets
% numImages = length(tgt.id);
% numImages = 1;

% Get each set of targets
tgt.set = find(tgt.id==1);
numImages = length(tgt.set);
%numImages = 3;

for k = 1:numImages
%     tgt.set = find(tgt.id==k);
%     pwNum = tgt.pw(tgt.set,1); % PW for this set of targets
    pwNum = tgt.pw(tgt.set(k),1); % PW for this set of targets

    % Radar parameters
    c = physconst('LightSpeed');
    fs = radar.fs(pwNum);
    fc = radar.fc(pwNum);
    pw = radar.pw(pwNum);
    pri = radar.pri(pwNum);
    prf = radar.prf(pwNum);
    peakpower = radar.peakPwr(pwNum);
    txgain = radar.txGain(pwNum);
    rxgain = radar.rxGain(pwNum);
    wbBeamforming = radar.wb(pwNum);
    nPulses = radar.nPulses(pwNum);
    noiseFigure = radar.noiseFigure(pwNum);
    maxTgtRange = radar.maxTgtRng(pwNum);
    deltaRange = radar.deltaRange(pwNum);
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
    tgtVel(1,:) = tgt.dwellVel; % km/sec
    tgtrcs = db2pow(10)*tgt.rcs(tgt.set(k),1);

    % Apply target motion
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

        receiver = phased.ReceiverPreamp( ...
        'SampleRate',fs, ...
        'Gain',rxgain, ...
        'NoiseFigure',noiseFigure);
    %%
    % Loop over the pulses to create a data cube of 64 pulses. For each step of
    % the loop, move the target and propagate the signal. Then put the received signal
    % into the data cube. The data cube contains the received signal per pulse.

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
    tWin = taylorwin(nSamples,4,-140);

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

    % Determine the range bins
%         fasttime = unigrid(0,1/fs,1/prf,'[)');
%         rangebins = (physconst('LightSpeed')*fasttime)/2;
%         %
%         probfa = 1e-12;
%         NoiseBandwidth = 5e6/2;
%         npower = noisepow(NoiseBandwidth,...
%             receiver.NoiseFigure,receiver.ReferenceTemperature);
%         thresh = npwgnthresh(probfa,nPulses,'coherent');
%         thresh = sqrt(npower*db2pow(thresh));
%         [~,range_detect] = findpeaks(pulsint(abs(datacube),'coherent'),...
%             'MinPeakHeight',thresh,'SortStr','descend');
%     
%         range_estimate = rangebins(range_detect(5));
%         ts = datacube(range_detect(9),:).';
%         [Pxx,F] = periodogram(ts,[],1024,prf,'centered');
%         plot(F,10*log10(Pxx))
% 
%         plot(real(datacube))
% 
%         figure 
%         t = (0:nPulses*nSamples-1)/waveform.SampleRate;
%         y = abs(datacube(:,32,:));
%         plot(y(:));title('Range profile (IFFT)'); xlabel('Range (m)'); ylabel('Magnitude')

 
    %%
    % Create and display the range-Doppler image for 64 Doppler bins. The image
    % shows range vertically and speed horizontally. Use the linear FM waveform for
    % match filtering. The image is here is the range-Doppler map.

    %Create a range-Doppler response object.
    dopResponse = phased.RangeDopplerResponse('DopplerFFTLengthSource','Property', ...
        'DopplerFFTLength',4096, ...
        'SampleRate',fs,'DopplerOutput','Frequency');
% 
     matchingcoeff = getMatchedFilter(waveform);
%     filter = phased.MatchedFilter('Coefficients',getMatchedFilter(waveform));
%     x = waveform();
%     y = filter(x);
%     figure;
%     plot(single(real(y)))
%     xlabel('Samples')
%     ylabel('Amplitude')
%     title('Matched Filter Output');
     [Rngdopresp,Rnggrid,Dopgrid] = dopResponse(datacube,matchingcoeff); %

    posDopplerShift = prf/2;
    negDopplerShift = -(posDopplerShift);

    figure
    imagesc(Dopgrid/2,Rnggrid,mag2db(abs(Rngdopresp)));
    colormap(jet(256));
    colorbar
    caxis([-30, 30]);

    ylim([min(tgtPos(1,:))-deltaRange max(tgtPos(1,:))+deltaRange])
   % ylim([20e3 70e3])
    xlim([negDopplerShift/2 posDopplerShift/2])
    xlabel('Doppler Frequency (Hz)')
    ylabel('Range (m)')
    title('Single Target Range-Doppler Map');

    fprintf('Using PW%d\n',pwNum)
    tmpFile = fullfile(path,'images', 'tmpImage');

    % Write the file image output
    saveas(gcf,tmpFile,'png')
    %close  % Current figure
end % For loop number of Images
end % Function
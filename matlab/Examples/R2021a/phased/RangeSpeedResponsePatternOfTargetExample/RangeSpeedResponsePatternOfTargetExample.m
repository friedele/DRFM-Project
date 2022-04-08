%% Range-Speed Response Pattern of Target
% This example shows how to visualize the speed and range of a target in a
% pulsed radar system that uses a rectangular waveform.
%%
% Place an isotropic antenna element at the global origin
% _(0,0,0)_. Then, place a target with a nonfluctuating
% RCS of 1 square meter at _(5000,5000,10)_, which is
% approximately 7 km from the transmitter. Set the operating (carrier)
% frequency to 10 GHz. To simulate a monostatic radar, set the
% |InUseOutputPort| property on the transmitter to
% |true|. Calculate the range and angle from the
% transmitter to the target.

%%
%
antenna = phased.IsotropicAntennaElement(...
    'FrequencyRange',[5e9 15e9]);
transmitter = phased.Transmitter('Gain',20,'InUseOutputPort',true);
fc = 10e9;
target = phased.RadarTarget('Model','Nonfluctuating',...
    'MeanRCS',1,'OperatingFrequency',fc);
txloc = [0;0;0];
tgtloc = [5000;5000;10];
antennaplatform = phased.Platform('InitialPosition',txloc);
targetplatform = phased.Platform('InitialPosition',tgtloc);
[tgtrng,tgtang] = rangeangle(targetplatform.InitialPosition,...
    antennaplatform.InitialPosition);
%%
% Create a rectangular pulse waveform 2&mu;s in duration with a PRF of
% 10 kHz. Determine the maximum unambiguous range for the given PRF. Use
% the radar equation to determine the peak power
% required to detect a target. This target has an RCS of 1 square meter at
% the maximum unambiguous range for the transmitter operating frequency and
% gain. The SNR is based on a desired false-alarm rate of
% $1e^{-6}$ for a noncoherent detector.

waveform = phased.RectangularWaveform('PulseWidth',2e-6,...
    'OutputFormat','Pulses','PRF',1e4,'NumPulses',1);
c = physconst('LightSpeed');
maxrange = c/(2*waveform.PRF);
SNR = npwgnthresh(1e-6,1,'noncoherent');
lambda = c/target.OperatingFrequency;
maxrange = c/(2*waveform.PRF);
tau = waveform.PulseWidth;
Ts = 290;
dbterm = db2pow(SNR - 2*transmitter.Gain);
Pt = (4*pi)^3*physconst('Boltzmann')*Ts/tau/target.MeanRCS/lambda^2*maxrange^4*dbterm;


%%
% Set the peak transmit power to the value obtained from the radar equation.
transmitter.PeakPower = Pt;
%%
% Create radiator and collector objects that operate at 10 GHz. Create a
% free space path for the propagation of the pulse to and from the target.
% Then, create a receiver.
radiator = phased.Radiator(...
    'PropagationSpeed',c,...
    'OperatingFrequency',fc,'Sensor',antenna);
channel = phased.FreeSpace(...
    'PropagationSpeed',c,...
    'OperatingFrequency',fc,'TwoWayPropagation',false);
collector = phased.Collector(...
    'PropagationSpeed',c,...
    'OperatingFrequency',fc,'Sensor',antenna);
receiver = phased.ReceiverPreamp('NoiseFigure',0,...
    'EnableInputPort',true,'SeedSource','Property','Seed',2e3);

%%
% Propagate 25 pulses to and from the target. Collect the echoes at the
% receiver, and store them in a 25-column matrix named |rx_puls|.
numPulses = 25;
rx_puls = zeros(100,numPulses);
%%
% Simulation loop
for n = 1:numPulses
    %%
    % Generate waveform
    wf = waveform();
    %%
    % Transmit waveform
    [wf,txstatus] = transmitter(wf);
    %%
    % Radiate pulse toward the target
    wf = radiator(wf,tgtang);
    %%
    % Propagate pulse toward the target
    wf = channel(wf,txloc,tgtloc,[0;0;0],[0;0;0]);
    %%
    % Reflect it off the target
    wf = target(wf);
    %%
    % Propagate the pulse back to transmitter
    wf = channel(wf,tgtloc,txloc,[0;0;0],[0;0;0]);
    %%
    % Collect the echo
    wf = collector(wf,tgtang);
    %%
    % Receive the target echo
    rx_puls(:,n) = receiver(wf,~txstatus);
end
%%
% Create a range-Doppler response object that uses the matched filter
% approach. Configure this object to show radial speed rather than Doppler
% frequency. Use |plotResponse| to plot the range versus
% speed.
rangedoppler = phased.RangeDopplerResponse(...
    'RangeMethod','Matched Filter',...
    'PropagationSpeed',c,...
    'DopplerOutput','Speed','OperatingFrequency',fc);
plotResponse(rangedoppler,rx_puls,getMatchedFilter(waveform))
%%
% The plot shows the stationary target at a range of approximately 7000 m.

%%
% Copyright 2012 The MathWorks, Inc.
function helperslexBistaticParam
% This function helperslexBistaticParam is only in support of
% slexBistaticExample. It may be removed in a future release.

%   Copyright 2014 The MathWorks, Inc.

paramBistatic.Fs = 3e6;
paramBistatic.tau = 20/3e6;
paramBistatic.PRF = 3125;
paramBistatic.bw = 3e6;
paramBistatic.ppow = 2000;
paramBistatic.RadarTxPos = [0;0;0];
paramBistatic.RadarTxVel = [0;0;0];
paramBistatic.Fc = 3e8;
paramBistatic.MaxRange = 40e3;
paramBistatic.TargetPos = [15000 35000;1000 -1000;500 100];
paramBistatic.TargetVel = [100 -160;100 0;0 -50];
paramBistatic.RadarRxPos = [20000;1000;100];
paramBistatic.RadarRxVel = [0;20;0];
paramBistatic.RCS = [2.5 4];
paramBistatic.pspeed = physconst('lightspeed');

myLFM = phased.LinearFMWaveform('SampleRate',paramBistatic.Fs,...
    'PulseWidth',paramBistatic.tau,...
    'PRF',paramBistatic.PRF,...
    'SweepBandwidth',paramBistatic.bw);
paramBistatic.coeff = getMatchedFilter(myLFM);
paramBistatic.rngoffset = numel(paramBistatic.coeff)*...
    paramBistatic.pspeed/paramBistatic.Fs;

clear myLFM;

assignin('base','paramBistatic',paramBistatic)
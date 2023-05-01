% waveform = phased.LinearFMWaveform('PulseWidth',100e-6,...
%     'SweepBandwidth',2e5,'PRF',1e3);

clear
path = 'C:\Users\friedele\Repos\DRFM';
radarFile = fullfile(path,'inputs','radar.xlsx');
[radarOpt,radar] = getParameters(radarFile,"radar");

pwNum = 6;

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

fs = 0.25e6;
waveform = phased.LinearFMWaveform('PulseWidth', pw, ...
    'PRF',prf,'SweepBandwidth',fs/2);
wav = waveform();

[afmag_lfm,delay_lfm,doppler_lfm] = ambgfun(wav,...
    waveform.SampleRate,waveform.PRF);
surf(delay_lfm*1e6,doppler_lfm/1e3,afmag_lfm,...
    'LineStyle','none')
axis tight
grid on
view([140,35])
colorbar
xlabel('Delay \tau (\mus)')
ylabel('Doppler f_d (kHz)')
title('Linear FM Pulse Waveform Ambiguity Function')
% 
ambgfun(wav,fs,waveform.PRF,'Cut','Doppler');
% deltav_lfm = dop2speed(20e3,c/fc)
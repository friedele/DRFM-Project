classdef arrayWaveform < handle
    % Superclass for DARC Waveform generation. In the future, this will
    % encompass Range Doppler (LFM), Synthetic Wideband, and Hyperbolic
    % Frequency Modulation. Note that the Synthetic Wideband waveform is
    % "build up" from the LFM.
    
    properties(Access=public)
        dutyCycle = 0.16;
        PRT = 0.0063;
        PRF = 160;
        Ts = 100e-6;
        fs = 250e6;
        pulseWidth = 1e-3;
        fCenter = 3e8; % Center Frequency (Hz)
        fSweep = 50e6;
        phiNaught = 0;
        fStep;
        numSteps = 'N/A';
        signal;
        waveformType = 'Range-Doppler';
        waveformLib = {};
        numPulses = 40;
        t;
        waveform;
    end
    
    properties(SetAccess=immutable)
        speedLight = 299792458; % Speed of Light
    end
    
    properties(Access=private)
        waveforms = {};
        waveformDef;
    end
    
    
    methods
        
        % DARC Waveform Constructor
        function o = arrayWaveform()
        end
        
        % Create a waveform library containing all of the core DARC
        % waveforms
        
        function rangeDoppler(o, fs, prf, pulseWidth, fSweep, numPulses)
            o.fs = fs;
            o.Ts = 1/fs;
            o.PRF = prf;
            o.PRT = 1/prf;
            o.pulseWidth = pulseWidth;
            o.fSweep = fSweep;
            o.dutyCycle = pulseWidth * prf;
            o.numSteps = 1;
            o.fCenter = fSweep/2;
            o.numPulses = numPulses;
            
            waveformDefStr = ["'SampleRate', fs, 'PulseWidth', pulseWidth', "...
                "'PRF', prf, 'SweepBandwidth', fSweep, 'NumPulses', " numPulses];
            tmp = string(strcat(waveformDefStr{2}, waveformDefStr{3}));
            waveformDefStr = string([waveformDefStr{1}; tmp])';
            
            functionCallStr = 'phased.LinearFMWaveform';
            
            o.waveformDef = strcat(waveformDefStr{1}, waveformDefStr{2});
            o.waveform = eval(strcat([functionCallStr, '(', ...
                waveformDefStr{1}, waveformDefStr{2}, ')']));
            o.signal = o.waveform();
            o.waveformType = 'Range-Doppler';
        end
        
        function plotSignal(o, pulseNum)
            if ~exist('pulseNum', 'var')
                pulseNum = 1;
            end
            o.getSignal(pulseNum);
            xlabel('Time (seconds)')
            ylabel('Amplitude')
            pltTitleStr = [o.waveformType ' Waveform'];
            title(pltTitleStr);
            plot(o.t, real(o.signal))
            hold on;
            plot(o.t, imag(o.signal))
            grid minor
        end
        
        function addWaveform(o, waveformDef)
            % % Currently incomplete. In the future, this will allow you to
            % archive waveform keys for a given transmitter without needing
            % to reconfigure the waveform object.
            o.waveforms = cat(o.waveforms, waveformDef);
            o.waveformLib = phased.PulseWaveformLibrary('WaveformSpecification', ...
                o.waveforms);
        end
        
    end
    
    methods(Access=public)
        % Helper functions to construct waveforms. To keep class speedy,
        % this must be (optionally) called by the user.
        function getSignal(o, pulseNum)
            if ~exist('pulseNum', 'var')
                pulseNum = 1;
            end
            if strcmp(o.waveformType, 'Range-Doppler')
                o.signal = step(o.waveform);
                nSamp = size(o.signal, 1);
                o.t = (0:(nSamp-1))./o.fs;
            elseif strcmp(o.waveformType, 'Hyperbolic')
                o.HyperbolicFM(pulseNum);
                nSamp = length(o.signal);
                o.t = (0:(nSamp-1))./o.fs;
            elseif strcmp(o.waveformType, 'Synthetic-Wideband')
                
                % Assemble the synthetic wideband signal
                wavfull = [];
                for k = 1:o.numSteps
                    wav = step(o.waveform);
                    wavfull = [wavfull; wav];
                end
                o.signal = wavfull;
                nSamp = size(o.signal, 1);
                o.t = (0:(nSamp-1))./o.fs;
            end
        end

        function HyperbolicFM(o, pulseNum)
            PRI = o.PRT;
            bandwidth = o.fSweep;
            fs_ = o.fs;
            % To do: move to static method so that this is not recalculated
            % for each pulse
            tSamp = ((1:o.numPulses+1) - 1)*PRI;
            f1 = o.fCenter - bandwidth/2;
            f2 = o.fCenter + bandwidth/2;
            k = (f1 - f2) / (f1*f2*tSamp(o.numPulses));
            mu  = -k *( f1 ./ (1 + f1*k*tSamp ) ).^2;   % Local chirp-slope for each sub-pulse to match the slope of the HFM curve
            avgEndMu = ( (1-o.dutyCycle)*mu(o.numPulses) + o.dutyCycle*mu(o.numPulses+1) );
            f2 = o.fCenter + bandwidth/2 - avgEndMu*o.pulseWidth + 0*63075.e0;
            k = (f1 - f2) / (f1*f2*tSamp(o.numPulses));
            mu  = -k *( f1 ./ (1 + f1*k*tSamp ) ).^2;
            f = f1./(1 + k*f1*tSamp ); % f0 is zeroed out since this is a baseband signal

            % Compute the number of samples required per PRI
            numSamplesPRI = floor(PRI*fs_);
            numSamplesPRI = numSamplesPRI + mod(numSamplesPRI+1, 2); % Adjust to ensure it is an odd number
            
            % Make slight adjustment to sample rate to make sure WF template is sampled with an odd number of samples and the
            % temporal duration of the pulse remains fixed with the exact original value, T. beta = (NumSamplesPerSubPulse*NCode) / T
            fs_ = numSamplesPRI / PRI;
            t_ = o.pulseWidth*linspace(0, 1, o.pulseWidth*fs_);    % Time-samples over one sub-pulsewidth
            numSamples = length(t_);
            
            % Compute the Baseband signal with carrier removed for a target with no Doppler Shift
            
            basebandSig = zeros(1, numSamplesPRI);
            basebandSig(1:numSamples) = exp( 1i*2*pi*f(pulseNum).*t_(1:numSamples) ).*exp( 1i*2*pi*(.5*mu(pulseNum)*(t_.^2) + o.phiNaught));
            o.signal = basebandSig';
        end
        

    end
    
end




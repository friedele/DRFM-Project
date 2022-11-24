classdef arrayTransmitter < arrayTxArray
    % Class to define the properties of the array Transmitter(s). This
    % object supports phased radar configurations. Inheritance from
    % the arrayPlatform, Antenna, and Waveform classes allows for the
    % placement of discrete transmitter elements. This paradigm allows for
    % individual control and orientation of the tx elements. This class
    % also allows for time-delay Tx beamforming. Includes functionality of
    % the radiator.
    
    % REFERENCE: https://www.mathworks.com/help/phased/ug/transmitter.html
    
    properties
        peakPower = 5000; % Watts
        gain = 130; % dB
        lossFactor = 0; % dB
        tx;
        txDuty;
        inUseOutputPort = true;
        coherentOnTransmit = 1;
        radiatedSignal;
    end

    properties(Access=private)
        TxDef;
        phaseNoiseOutputPort;
    end
    
    methods
        function o = arrayTransmitter()
        end
        
        
        function txSignal(o)
            
            TransmitterStr = ["'PeakPower', o.peakPower, 'Gain', o.gain', "...
                "'LossFactor', o.lossFactor, 'CoherentOnTransmit', o.coherentOnTransmit, "...
                "'InUseOutputPort', o.inUseOutputPort'"];
            functionCallStr = 'phased.Transmitter';

            o.TxDef = strcat(TransmitterStr{1}, TransmitterStr{2}, TransmitterStr{3});
            
            txObj = eval(strcat([functionCallStr, '(', ...
                TransmitterStr{1}, TransmitterStr{2}, TransmitterStr{3} ')']));
            
            [o.tx, o.txDuty] = txObj(o.signal);
            
        end
        
        function getRadiator(o)
            % Constructor for wideband and narrowband radiators. In order
            % to accomodate the sparse array configuration, each transmit
            % element is treated as a subarray. This allows us to
            % implement independent electronic steering for each element
            % while accomodating both narrowband and wideband transmit
            % waveforms.
            
            if ~strcmpi(o.waveformType, 'synthetic-wideband') || ...
                    ~strcmpi(o.waveformType, 'hfm')
                o.radiator = phased.WidebandRadiator('Sensor', o.antennaArray, ...
                    'CarrierFrequency', o.fCarrier, 'PropagationSpeed', o.c);
            else
                % Use the narrowband radiator
                o.radiator = phased.Radiator('Sensor',o.antennaArray,'PropagationSpeed', o.c, ...
                'OperatingFrequency',o.fCarrier);
            end
        end
        
        function radiateSignal(o)
            % Radiate the signal in the desired azimuth elevation. Changing 
            az = mean(o.antennaArray.ElementNormal(1,:));
            el = mean(o.antennaArray.ElementNormal(2,:));
            o.radiatedSignal = o.radiator(o.tx, [az; el]);
        end

        
        function plotTx(o)
            subplot(2, 1, 1)
            plot(o.t, real(o.tx));
            axis tight
            grid on
            ylabel('Amplitude')
            title('Transmitter Output (real part) - One PRI')
            subplot(2, 1, 2)
            plot(o.t, o.txDuty)
            axis([0 o.t(end) 0 1.5]);
            xlabel('Seconds')
            grid on
            ylabel('Off-On Status')
            set(gca, 'ytick', [0 1])
            title('Transmitter Status')
        end

    end

end
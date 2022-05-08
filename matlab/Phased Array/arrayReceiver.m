classdef arrayReceiver < arrayRxArray
    
    properties
        gain = 20;
        lossFactor = 0;
        EnableInputPort = true;
        collector;
        rxAmplifiedSignal;
        wideband = true; 
        rxDishReturns;
    end
    
    
    methods
        %% Method for amplifying the receive signal
        function applyPreamp(o, rxInputSig)
            receiver = phased.ReceiverPreamp('Gain', o.gain, 'LossFactor', ...
                o.lossFactor);
            o.rxAmplifiedSignal = receiver(rxInputSig);
        end
        
        %% Set up the collectors for energy impinging on the array.
        % Use methods from DARCPlatform to get the incident angles on a
        % per-element basis (beamforming is performed in the frequency
        % domain, so "the signal arrives how it arrives"
        
        function getCollector(o, fs)
           
            % Determine if we are using the wideband or narrowband
            % collector
            
            if o.wideband
                o.collector = phased.WidebandCollector('Sensor', o.antennaArray,...,
                    'Wavefront', 'Unspecified', ...
                    'CarrierFrequency', o.fCarrier, 'SampleRate', fs);
            else
                % Narrowband collector
                o.collector = phased.Collector('Sensor', o.antennaArray,...
                    'Wavefront', 'Unspecified', ...
                    'OperatingFrequency', o.fCarrier); % Removed sample rate, its not part of this class.
            end
        end
        
        function getRxElementReturns(o, incidentAngles)
           % Get the returns for each rx pedestal. Use the platform class'
           % getLocalRangeAngle method to get the incidentAngles array.
           o.rxDishReturns = o.collector(o.rxAmplifiedSignal, incidentAngles);

        end
    end

end
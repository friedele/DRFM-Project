classdef arrayNoiseJammer < arrayReceiver
    % Class for various noise jammers. This class utilizes
    % the createTarget method in the arrayPlatform class.  The noise jammer
    % will be attached to the radar target.
    properties(Access=public)
        erp = 1e3;  % Effective radiated power (watts)
        fs;
        jammerChannel;
        jammerCollector;
        jammerPlatform;
        jammerPos;
        rxAngleToJammer;
        rxRangeToJammer;
        rxJammerReturns;
        rxPos; % Position in global frame
        rxJammerWaveProp;
        samplesJam = 6250;  % Number of jammer noisesamples
        signalJam;
        seed = 777;
        txPos; % Position in global frame
    end
    
    methods
        % DARC Noise Jammer Constructor
        function o = DARCNoiseJammer()
        end
        
        % Generates a complex white Gaussian noise jamming target signal.
        % Power is spread over a broad enough range of frequencies to
        % blanket all the radar frequencies.
        function barrageJammer(o)
            bJammer = phased.BarrageJammer('ERP',o.erp, 'SamplesPerFrame', o.samplesJam);
            o.signalJam = step(bJammer);
        end
        
        % Generates a complex white Gaussian noise jamming target signal at
        % particular radar frequencies. If properly timed should produce
        % false targets
        function spotJammer(o)
            spotJammer = phased.BarrageJammer('ERP',o.erp, 'SamplesPerFrame', o.samplesJam);
            o.signalJam = step(spotJammer);
        end
        
        %  Generates a complex white Gaussian noise jamming target signal
        %  as a burst of noise as it sweeps through its frequencies.  The
        %  burst will be for a short amount of time.
        function sweepJammer(o)
            sweepJammer = phased.BarrageJammer('ERP',o.erp, 'SamplesPerFrame', o.samplesJam);
            o.signalJam = step(sweepJammer);
        end
        
        function getJammerRangeAngle(o, receiverObj, jammerPos)
            localPosRx = receiverObj.antennaArray.ElementPosition;
            if size(o.rxPos, 1) == 1
                o.rxPos = o.rxPos';
            end
            globalPosRx = local2globalcoord(localPosRx, 'rr', o.rxPos);
            
            rxToJammerRange_LCS = NaN(1, size(globalPosRx, 2));
            rxToJammerAng_LCS = NaN(2, size(globalPosRx, 2));
            
            for k = 1:size(globalPosRx, 2)
                [rxToJammerRange_LCS(:,k), rxToJammerAng_LCS(:,k)] = rangeangle(jammerPos(:), globalPosRx(:,k), eye(3)); %OK
            end
            
            o.rxRangeToJammer = rxToJammerRange_LCS;
            o.rxAngleToJammer = rxToJammerAng_LCS;
        end
        
        function propagateJammer(o, receiverObj, jammerObj)
            
            % Get the global location of the receiver elements
            localPosRx = receiverObj.antennaArray.ElementPosition;
            
            % Convert the local coordinates to global coordinates per element
            if size(o.txPos, 1) == 1
                o.txPos = o.txPos';
            end
            
            if size(o.rxPos, 1) == 1
                o.rxPos = o.rxPos';
            end
            
            globalPosRx = local2globalcoord(localPosRx, 'rr', o.rxPos);
            o.rxJammerWaveProp = NaN(length(jammerObj.signalJam), size(receiverObj.antennaArray.ElementPosition, 2));
            
            for k = 1:size(receiverObj.antennaArray.ElementPosition, 2)
                o.rxJammerWaveProp(:,k) = o.jammerChannel(jammerObj.signalJam, jammerObj.jammerPos', globalPosRx(:,k), [0;0;0],[0;0;0]);
            end
        end
        
        function createJammerPlatform(o, jammerPos)
            % Define the receiver platform
            o.jammerPos = [jammerPos(1) jammerPos(2) jammerPos(3)];
            o.jammerPlatform = phased.Platform('InitialPosition', [jammerPos(1); jammerPos(2); jammerPos(3)], ...
                'Velocity', [0;0;0], 'OrientationAxesOutputPort', true);
        end
        
        function createJammerChannel(o, fs, fCarrier)
            o.fs = fs;
            o.jammerChannel = phased.FreeSpace('TwoWayPropagation',false,...
                'SampleRate',o.fs,'OperatingFrequency', fCarrier);
        end
        
        function jammerCollection(o)
                % Narrowband collector
                o.jammerCollector = phased.Collector('Sensor', o.antennaArray,...
                    'Wavefront', 'Unspecified', ...
                    'OperatingFrequency', o.fCarrier); 
        end
        
        function getJammerElementReturns(o, incidentAngles)
            % Get the returns for each rx pedestal. Use the platform class
            % getLocalRangeAngle method to get the incidentAngles array.
            o.rxJammerReturns = o.jammerCollector(o.rxAmplifiedSignal, incidentAngles);
        end
        
    end
end


classdef arrayPlatform < handle
    % Class to define the coordinate frame
    properties
        fCarrier = 3.3e9;       
        meanRCS;
        fs;
        tgtPos; % Position in global frame
        tgtVel;
        tgtAccel;
        tgtOri;
        txPos; % Position in global frame
        rxPos; % Position in global frame      
        txPlatform;
        rxPlatform;
        tgtPlatform;
        radarTarget;
        tgtRCS;
        txWaveProp;
        txRangeToTarget;
        txAngleToTarget;
        rxRangeToTarget;
        rxAngleToTarget;
        txChannel;
        rxChannel;
        rxWaveProp;
        swerlingCase = 2;
    end
    
    methods
        
        %% Methods for creating transmitter and receiver platforms
        
        function createTxPlatform(o, txPos)
            % Define the transmitter platform
            o.txPos = [txPos(1) txPos(2) txPos(3)];
            o.txPlatform = phased.Platform('InitialPosition', [txPos(1); txPos(2); txPos(3)], ...
                'Velocity', [0;0;0], 'OrientationAxesOutputPort', true);
        end
        
        function createRxPlatform(o, rxPos)
            % Define the receiver platform
            o.rxPos = [rxPos(1) rxPos(2) rxPos(3)];
            o.rxPlatform = phased.Platform('InitialPosition', [rxPos(1); rxPos(2); rxPos(3)], ...
                'Velocity', [0;0;0], 'OrientationAxesOutputPort', true);
        end
        
        %% Methods for creating the radar target. For now, this only supports one target.
        % This will be updated to support multiple targets in the future.
        
        function createTarget(o, tgtPos, tgtVel, tgtAccel, tgtOri)
            o.tgtPos = [tgtPos(1) tgtPos(2) tgtPos(3)];
            o.tgtVel = [tgtVel(1) tgtVel(2) tgtVel(3)];
            o.tgtAccel = [tgtAccel(1) tgtAccel(2) tgtAccel(3)];
            o.tgtOri = tgtOri;
            
            o.tgtPlatform = phased.Platform('InitialPosition', [o.tgtPos(1); o.tgtPos(2); o.tgtPos(3)], ...
                'MotionModel', 'Acceleration', 'InitialVelocity', [o.tgtVel(1); o.tgtVel(2); o.tgtVel(3)], ...
                'Acceleration', [o.tgtAccel(1); o.tgtAccel(2); o.tgtAccel(3)], ...
                'InitialOrientationAxes', [o.tgtOri], 'OrientationAxesOutputPort', true);
            if o.swerlingCase == 2
                tgtmodel = 'Swerling2';
            else
                tgtmodel = 'Swerling1';
            end
            o.radarTarget = phased.RadarTarget('OperatingFrequency', o.fCarrier, ...
                'Model', tgtmodel, 'MeanRCS', o.meanRCS);
        end
        
        %% Channel Models. Support for wideband, narrowband and jammer propagation.
        
        function createNarrowbandChannels(o, fs, maxDistance, maxNumSamples)
            o.fs = fs;
            o.txChannel = phased.FreeSpace('OperatingFrequency',o.fCarrier,...
                'TwoWayPropagation',false, 'SampleRate', o.fs);  % Tx
            o.rxChannel = phased.FreeSpace('OperatingFrequency', o.fCarrier,...
                'TwoWayPropagation',false, 'SampleRate', o.fs);  % Rx
        end
        
        function createWidebandChannels(o, fs, maxDistance, maxNumSamples)
            o.fs = fs;
            o.txChannel = phased.WidebandFreeSpace('OperatingFrequency', o.fCarrier, ...
                'TwoWayPropagation', false, 'SampleRate', o.fs);
            o.rxChannel = phased.WidebandFreeSpace('OperatingFrequency', o.fCarrier, ...
                'TwoWayPropagation', false, 'SampleRate', o.fs);
        end

        function propagateTx(o, txRad)
            % Propagate the waveform from the transmitter to the target
            o.txWaveProp = o.txChannel(txRad, o.txPos, o.tgtPos, [0; 0; 0], o.tgtVel); %OK
        end
        
        function propagateRx(o, receiverObj)
            % Get the RCS of the target. Reflect energy back to the
            % receiver.
            wavreflect = o.radarTarget(o.txWaveProp, true);
            % Propagate the waveform from the target to each individual rx
            % element
            
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
            o.rxWaveProp = NaN(length(o.txWaveProp), size(receiverObj.antennaArray.ElementPosition, 2));
            
            for k = 1:size(receiverObj.antennaArray.ElementPosition, 2)
                o.rxWaveProp(:,k) = o.rxChannel(wavreflect, o.tgtPos, globalPosRx(:,k), o.tgtVel,[0;0;0]);
            end
            
            %o.rxWaveProp = o.rxChannel(wavreflect, o.tgtPos,o.rxPos,o.tgtVel,0); %OK
        end

        
        function updateTarget(o, dt)
            [o.tgtPos, o.tgtVel, o.tgtAccel] = o.tgtPlatform(dt);
        end
        
        
        function getLocalRangeAngle(o, transmitterObj, receiverObj)
           % Compute the range and angle from all array elements to the target.
           % This method is intended to be used in order to update the
           % elementNormal component for all of the elements. As such, this
           % computation is done w.r.t. the local frame.
           
           % Because of the directivity of the elements, the only way to
           % ensure that there is enough energy in the phase center of the
           % virtual array is to determine the angle between the target and
           % each tx/rx element. These angles are then used to adjust the
           % ElementNormal of each element. This scheme makes it so that
           % the tx and rx array automatically track the target.
            
           %% Step 1, extract the element locations in the local frame.
           localPosTx = transmitterObj.antennaArray.ElementPosition;
           localPosRx = receiverObj.antennaArray.ElementPosition;
           
           %% Step 2, Convert the local coordinates to global coordinates per element
           if size(o.txPos, 1) == 1
               o.txPos = o.txPos';
           end
           
           if size(o.rxPos, 1) == 1
               o.rxPos = o.rxPos';
           end
           
           globalPosTx = local2globalcoord(localPosTx, 'rr', o.txPos);
           globalPosRx = local2globalcoord(localPosRx, 'rr', o.rxPos);
           
           %% Step 3, Compute the angle between each element and the target, with w.r.t. the local coordinate system
           if size(o.tgtPos, 1) == 1
               o.tgtPos = o.tgtPos';
           end
           
           txToTgtRange_LCS = NaN(1, size(globalPosTx, 2));
           txToTgtAng_LCS = NaN(2, size(globalPosTx, 2));
           
           for k = 1:size(globalPosTx, 2)
               [txToTgtRange_LCS(k), txToTgtAng_LCS(:,k)] = rangeangle(o.tgtPos, globalPosTx(:,k), eye(3)); %OK
           end
           
           %% Step 4, Repeat the above for each receive element

           rxToTgtRange_LCS = NaN(1, size(globalPosRx, 2));
           rxToTgtAng_LCS = NaN(2, size(globalPosRx, 2));
           
           for k = 1:size(globalPosRx, 2)
               [rxToTgtRange_LCS(k), rxToTgtAng_LCS(:,k)] = rangeangle(o.tgtPos, globalPosRx(:,k), eye(3)); %OK
           end
           
           o.txRangeToTarget = txToTgtRange_LCS;
           o.txAngleToTarget = txToTgtAng_LCS;
           o.rxRangeToTarget = rxToTgtRange_LCS;
           o.rxAngleToTarget = rxToTgtAng_LCS;
           
        end

    end

end





classdef arrayTxArray < arrayWaveform
    % Defines the antenna and radiator configuration for the
    % DARCTransmitter. The carrier frequency is given by fcenter/2.
    
    % NOTE: This DARCArray class defines the geometry using the Phased
    % Array toolbox. In order to "steer" these elements, we need to
    % manually reorient the elements.
    
    % For now, this ignores subarraying. This will be needed if we want to
    % point our beams in N distinct directions (required for the radiator).
    
    % Once (or while) the elements are mechanically slewed to a setpoint, a
    % steering vector is computed in order to place the beam from the
    % collective array into a specified azimuth-elevation sector. For now,
    % this is done by computing a set of phase delays.
    
    % In the future, all DARC classes will use getters and setters to
    % automatically update relevant attributes as waveforms are changed
    
    
    properties(SetAccess=immutable)
        % All DARC Antennas are parabolic reflectors. MATLAB 2018 does not
        % have this antenna model. Using a collection of dipole antennas to
        % stand-in for now.
        c = 299792458; % Speed of light
    end
    
    properties
        % Add support for polarization?
        antennaArray; % Object
        numElements=6;
        elementSpacing;
        lambda;
        elementType = 'darc';
        element; % 1x1 element object
        radiator;
        maxSlew = 5; % Degrees per second
        xPosition = [0 50 100 150 200 250];
        yPosition = [0 10 -10 0 10 -20];
        zPosition = [0 0 0 0 0 0];
        elNormAz = -111.961 % True North 360 degree clockwise azimuth
        elNormEl = 055.5084743065231 % 0 points at horizon, 90 points at zenith
        txWeights;
        fCarrier = 3.3e9;
    end
    
    methods(Access=public)
        function o = arrayTxArray()
        end
        
        %% Methods for specifying array configuration
        
        function createTxArray(o)
            % Define a discrete antenna element (temporarily dipole)
            if strcmpi(o.elementType, 'dipole')
                o.element = phased.ShortDipoleAntennaElement('AxisDirection', 'Z');
                o.lambda = o.c/fCenter;
                o.elementSpacing = o.lambda/2;
                o.antennaArray = phased.ULA(o.numElements, o.elementSpacing, ...
                    'Element', o.element);
                
            elseif strcmpi(o.elementType, 'darc')
                % Custom DARC Element (TBD) - temp pattern for now
                o.element = load('C:\Users\j23091\MATLAB\Master_Repos\DARC_M&S\DBF\DigitalBeamForming\Algorithms\JHU_Antenna\elementCircularApertureHD.mat', 'antenna');
                o.antennaArray = phased.ConformalArray;
                o.antennaArray.Element = o.element.antenna;
                o.antennaArray.ElementNormal = [o.elNormAz; o.elNormEl];
            end
        end
        
        %% Methods for electronic and mechanical steering
        
        function slewElements(o, AzEl)
            % Simulates the effect of mechanical steering of individual
            % dishes. Manual slewing is accomplished by changing the
            % ElementNormal values of the defined conformal array. Recommend
            % using the showArray method to verify element-level indexing
            
            o.antennaArray.ElementNormal = AzEl;
        end
        
        function getPhaseSteerWeights(o)
            % Point the Array using Conventional  Beamforming
            % (Phase based). This is intended for transmit array steering.
            steerVec = phased.SteeringVector('SensorArray', o.antennaArray, ...
                'IncludeElementResponse', true, 'PropagationSpeed', o.c);
            
            % Due to the narrow width of the beam for the DARC tx array, we
            % will set the desired azimuth and elevation for phase steering
            % to the phase center of the array.
            az = mean(o.antennaArray.ElementNormal(1,:));
            el = mean(o.antennaArray.ElementNormal(2,:));
            
            sv = steerVec(o.fCarrier, [az; el]);
            o.txWeights = sv;
            
        end
        
        function nullSteerWeights(o, nullAz, nullEl)
            % Relevant MATLAB Example
            % https://www.mathworks.com/help/phased/examples/array-pattern-synthesis.html
            % Weights can either be applied using the Collector/Radiator
            % weight method, or by changing the taper property in the
            % definition of the array. Assuming that these methods are
            % equivalent, we will perform these modifications in the
            % collector and radiator. Will need to do further analysis to
            % determine the beamwidth so that reasonable nulling az els are
            % provided.
            
            pointAz = o.antennaArray.ElementNormal(1);
            pointEl = o.antennaArray.ElementNormal(2);
            
            % Calculate the steering vector for null direction
            wNull = phased.SteeringVector('SensorArray', o.antennaArray, ...
                'IncludeElementResponse', true);
            wNull = wNull(o.fCarrier, [nullAz; nullEl]);
            
            % Calculate the steering vector for the look direction
            wLook = phased.SteeringVector('SensorArray', o.antennaArray, ...
                'IncludeElementResponse', true);
            wLook = wLook(o.fCarrier, [pointAz, pointEl]);
            
            % Compute the response of desired steering at null direction
            rNull = wNull'*wLook/(wNull'*wNull);
            
            % Sidelobe canceler - remove the response in null direction
            
            o.txWeights = wLook-wNull*rNull;
            
        end
        
        
        %% Methods to show the array geometry
        
        function showArray(o)
            viewArray(o.antennaArray, 'ShowIndex', 'All', 'ShowNormals', true)
        end
        
        %% Methods to display the antenna pattern
        function showPattern(o)
            azCuts = -180:180;
            elCuts = -90:90;
            pattern(o.antennaArray, o.fCarrier, azCuts, elCuts, 'PropagationSpeed',...
                o.c, 'CoordinateSystem', 'polar', 'Type', 'powerdb', 'Normalize',...
                true)
        end
        
        function showElCutPattern(o)
            figure();
            elCuts = linspace(-90, 90, 8000);
            azCut = 0;
            pattern(o.antennaArray, o.fCarrier, azCut, elCuts, 'PropagationSpeed',...
                o.c, 'CoordinateSystem', 'polar', 'Type', 'powerdb', 'Normalize', ...
                true)
            figure();
            pattern(o.antennaArray, o.fCarrier, azCut, elCuts, 'PropagationSpeed', ...
                o.c, 'CoordinateSystem', 'rectangular', 'Type', 'powerdb', 'Normalize', ...
                true)
        end
        
        function showAzCutPattern(o)
            figure();
            azCuts = linspace(-180, 180, 20000);
            elCut = 0;
            pattern(o.antennaArray, o.fCarrier, azCuts, elCut, 'PropagationSpeed',...
                o.c, 'CoordinateSystem', 'polar', 'Type', 'powerdb', 'Normalize',...
                true)
            figure();
            pattern(o.antennaArray, o.fCarrier, azCuts, elCut, 'PropagationSpeed', ...
                o.c, 'CoordinateSystem', 'rectangular', 'Type', 'powerdb', 'Normalize', ...
                true)
            
        end
        
        function showSteeringPattern(o)
            % Show the effect of steering on the beam pattern. This will
            % only show the pattern if the elements are oriented such that
            % the element azimuth and elevation are at 0 degrees.
            figure();
            subplot(211)
            pattern(o.antennaArray, o.fCarrier, linspace(-180, 180, 20000), linspace(-90, 90, 10000), 'CoordinateSystem', 'rectangular', ...
                'PropagationSpeed', o.c, 'Type', 'powerdb')
            title('Without Steering')
            
            subplot(212)
            pattern(o.antennaArray, o.fCarrier, linspace(-180, 180, 20000), linspace(-90, 90, 10000), 'CoordinateSystem', 'rectangular', ...
                'PropagationSpeed', o.c, 'Type', 'powerdb', 'Weights', o.txWeights)
            title('With Steering')
        end
        
    end
    
    
end
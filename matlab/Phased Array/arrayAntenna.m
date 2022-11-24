classdef arrayAntenna < handle
    
    % Class for the Receive Array. Note that this class is defined
    % independently of the arrayTxArray superclass (and does not use inheritance)
    % to avoid class bloat.
    
    properties(SetAccess=immutable)
        c = 299792458; % Speed of light
    end
    
    
    properties
        % Add support for polarization?
        antennaArray; % Object
        numElements=1000;
        element; % 1x1 element object
        maxSlew = 5; % Degrees per second
        xPosition = [0 30 40 150 200 250];
        yPosition = [0 10 -10 0 10 -20];
        zPosition = [0 0 0 0 0 0];
        elNormAz = -111.961 % True North 360 degree clockwise azimuth
        elNormEl = 055.5084743065231 % 0 points at horizon, 90 points at zenith
        fCarrier = 3.3e9;        
    end
   
    methods(Access=public)
        function o = arrayAntenna()
        end
        
        function createArrayAntenna(o)
            % Define a discrete antenna element (temporarily dipole)
          
                % Custom DARC Element (TBD) - temp pattern for now
                o.element = load('C:\Users\j23091\MATLAB\Master_Repos\DARC_M&S\DBF\DigitalBeamForming\Algorithms\JHU_Antenna\elementCircularApertureHD.mat', 'antenna');
                o.antennaArray = phased.ConformalArray;
                o.antennaArray.Element = o.element.antenna;
                o.antennaArray.ElementPosition = ...
                    [o.xPosition; o.yPosition; o.zPosition];
                o.antennaArray.ElementNormal = [o.elNormAz; o.elNormEl];  
        end

        function slewElements(o, AzEl)
            % Simulates the effect of mechanical steering of individual
            % dishes. Manual slewing is accomplished by changing the
            % ElementNormal values of the defined conformal array. Recommend
            % using the showArray method to verify element-level indexing
            
            % TO DO: Add check for slew rate
            o.antennaArray.ElementNormal = AzEl;
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


        function showArray(o)
            viewArray(o.antennaArray, 'ShowIndex', 'All', 'ShowNormals', true)
        end

    end
    
    
end
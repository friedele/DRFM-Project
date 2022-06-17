classdef DARCCentralBeamformer < handle
    
    % Class for the DARC Central Beamformer. This class contains methods
    % for performing signal processing operations specified by the JHU
    % Algorithm Description document.
    
    properties(SetAccess=immutable)
        c = 299792458; % Speed of light
    end
    
    properties
        fastTimePaddingFactor = 1;
        slowTimePaddingFactor = 1;
        PRF = 100;
        fs = 1e6;
        rangeGates;
        dopplerGates;
        rdMap;
        rangeOffset;
        fSweep;
        fCenter = 9e9;
        noiseFig = 1;
        refTemp = 290;
        numCalls = 1;
        timeOffset;
        rxPacketTimes;
        dynamicRange = 40;
        rangeResponse;
        waveformType = 'Range-Doppler';
        deltaF = 0;
        HFM = 0;
        tgtPos;
    end
    
    methods
        
        % DARC CBF Constructor
        function o = DARCCentralBeamformer()
        end
        
        function o = setParameters(x)
           o=x;
        end
        
        function rangeDopplerResponse(o, rxDishReturns, replica)
            % Method to perform pulse-by-pulse range profile calculation
            % followed by Doppler processing. All appropriate scaling is
            % performed
            
            % Take the FFT of the rxDishReturns matrix and the replica
            numCols = size(rxDishReturns, 1);
            numRows = size(rxDishReturns, 2);
            
            % Compute the Range Responses
            X = fft(rxDishReturns, numCols, 1);
            H = fft(replica, numCols, 1);
            Y = X.*H;
            y = ifft(Y, numCols*o.fastTimePaddingFactor, 1);
            
            % Compensate for the Matched Filter delay
            matchingdelay = o.fastTimePaddingFactor*size(replica,1)-1;
            y = buffer(y(matchingdelay+1:end),size(y,1));
            
            %% Create the axes for Range and Doppler
            fast_time_grid = unigrid(0,...
                1/(o.fs*o.fastTimePaddingFactor),1/o.PRF,'[)');
            nfftDoppler = size(rxDishReturns,2)*o.slowTimePaddingFactor;
            slow_time_grid = o.PRF*...
                (-nfftDoppler/2:nfftDoppler/2-1)/nfftDoppler;
            prop_speed = physconst('LightSpeed');
            range_gates = prop_speed*fast_time_grid/2;
            lambda = prop_speed/o.fCenter;
            
            % Perform Automatic Gain Control to compensate for
            % range-dependent losses
            tvg = phased.TimeVaryingGain(...
                'RangeLoss',2*fspl(range_gates,lambda),...
                'ReferenceLoss',2*fspl(prop_speed/(o.PRF*2),lambda));
            
        %    y = tvg(y);
            
            % Compute the Doppler response
            resp = fftshift(fft(y, o.slowTimePaddingFactor*numRows, 2),2);
            NoiseBandwidth = o.fs;
            noisepow = physconst('Boltzmann')*...
                systemp(o.noiseFig, o.refTemp) * NoiseBandwidth;
            mfgain = replica(:,1)'*replica(:,1);
            noisepower_proc = mfgain*noisepow;
            scaledMat = pow2db(abs(resp).^2/noisepower_proc);
            maxVal = max(max(scaledMat));
            % Apply a range offset proportional to account for the radar
            % unambiguous range
            o.rangeOffset = max(range_gates) * (o.numCalls - 1);
            o.timeOffset = max(range_gates)/physconst('lightspeed') * ...
                (o.numCalls - 1);
            o.rxPacketTimes = linspace(0, 1/o.PRF, ...
                size(rxDishReturns,2)) + o.rangeOffset;
            imagesc(slow_time_grid,(o.rangeOffset + ...
                range_gates)/1e3,pow2db(abs(resp).^2/noisepower_proc));
            xlabel('Doppler Frequency (Hz)'); ylabel('Range (km)');
            title('Range Doppler Map');
            if o.deltaF <= 0
                titleStr = ['Range-Doppler Response' ...
                    'f_{Sweep} = ' num2str(o.fSweep/1e6) 'MHz, PRF = ' num2str(o.PRF) ', tgtPos = ' ...
                    num2str(ceil(norm(o.tgtPos)/1e3)) 'km, numPulses = ' num2str(size(rxDishReturns,2))];
            else
                titleStr = ['Synthetic Wideband Range-Doppler Response' ...
                    'f_{Sweep} = ' num2str(o.fSweep/1e6) 'MHz, PRF = ' num2str(o.PRF) ', tgtPos = ' ...
                    num2str(ceil(norm(o.tgtPos)/1e3)) 'km, numPulses = ' num2str(size(rxDishReturns,2)), ...
                    ' \Delta_F = ' num2str(o.deltaF/1e3) 'KHz'];
            end
            title(titleStr)
            set(get(colorbar,'YLabel'),'String','SNR (dB)');
            set(gca,'YDir','normal');
            colormap('jet')
            caxis([maxVal-o.dynamicRange maxVal]);
            o.rangeGates = o.rangeOffset + range_gates;
            o.dopplerGates = slow_time_grid;
            o.rdMap = pow2db(abs(resp+eps).^2/noisepower_proc);
        end
        
        function rangeDopplerCuts(o, rxDishReturns, replica)
            % Create the Range-Doppler surface
            o.rangeDopplerResponse(rxDishReturns, replica);
            % Plot the range cut of the Range-Doppler response
            [~,zeroDopplerCutIdx] = min(abs(0 - o.dopplerGates));
            rangeCut = o.rdMap(:,zeroDopplerCutIdx);
            % Determine the theoretical range resolution
            %rangeRes = physconst('lightspeed')/(2*o.fSweep);
            [~,maxIdx] = max(rangeCut);
            % Add red vertical lines to the plot to show theoretical range
            % resolution relative to the peak
            %upperLimitRRes = o.rangeGates(maxIdx) + rangeRes;
            %lowerLimitRRes = o.rangeGates(maxIdx) - rangeRes;
            
            figure(); plot(o.rangeGates, rangeCut)
            %xline(lowerLimitRRes, 'r--')
            %xline(upperLimitRRes, 'r--', {'Theoretical Range Resolution Bounds'})
            xlabel('Range (km)')
            ylabel('Log Magnitude (dB)')
            %title('Range Cut of Range-Doppler Response Surface')
            if o.deltaF <= 0
                titleStr = ['Range-Doppler Range Response ' ...
                    'f_{Sweep} = ' num2str(o.fSweep/1e6) ' MHz, PRF = ' num2str(o.PRF) ', tgtPos = ' ...
                    num2str(ceil(norm(o.tgtPos)/1e3)) ' km, numPulses = ' num2str(size(rxDishReturns,2))];
            else
                titleStr = ['Synthetic Wideband Range Response ' ...
                    'f_{Sweep} = ' num2str(o.fSweep/1e6) ' MHz, PRF = ' num2str(o.PRF) ', tgtPos = ' ...
                    num2str(ceil(norm(o.tgtPos)/1e3)) ' km, numPulses = ' num2str(size(rxDishReturns,2)), ...
                    ' \Delta_F = ' num2str(o.deltaF/1e3) ' KHz'];
            end
            title(titleStr)
            grid minor
            % Compute the Doppler response using the calculated parameters
            %dopplerRes = o.PRF/size(rxDishReturns,2);
            dopplerCut = o.rdMap(maxIdx,:);
            % Find where the maximum value in the Doppler cut occurs
            %[~,maxDopplerIdx] = max(dopplerCut);
            %upperLimitDRes = o.dopplerGates(maxDopplerIdx) + dopplerRes;
            %lowerLimitDRes = o.dopplerGates(maxDopplerIdx) - dopplerRes;
            figure(); plot(o.dopplerGates, dopplerCut)
            %xline(lowerLimitDRes, 'r--')
            %xline(upperLimitDRes, 'r--', {'Theoretical Doppler Resolution Bounds'})
            xlabel('Doppler Frequency (Hz)')
            ylabel('Log Magnitude (dB)')
            if o.deltaF <= 0
                titleStr = ['Doppler Response ' ...
                    'f_{Sweep} = ' num2str(o.fSweep/1e6) ' MHz, PRF = ' num2str(o.PRF) ', tgtPos = ' ...
                    num2str(ceil(norm(o.tgtPos)/1e3)) ' km, numPulses = ' num2str(size(rxDishReturns,2))];
            else
                titleStr = ['Synthetic Wideband Doppler Response ' ...
                    'f_{Sweep} = ' num2str(o.fSweep/1e6) ' MHz, PRF = ' num2str(o.PRF) ', tgtPos = ' ...
                    num2str(ceil(norm(o.tgtPos)/1e3)) ' km, numPulses = ' num2str(size(rxDishReturns,2)), ...
                    ' \Delta_F = ' num2str(o.deltaF/1e3) ' KHz'];
            end
            %title('Doppler Cut of Range-Doppler Surface')
            title(titleStr)
            grid minor
            o.rangeResponse = rangeCut;
        end
        
        function rangeCut(o, rxDishReturns, replica)
              % Method to perform pulse-by-pulse range profile calculation
            % followed by Doppler processing. All appropriate scaling is
            % performed
            
            % Blank the first x ms of the receive window to account for 
            % range eclipsing
            
            rxDishReturns(1:length(replica), :) = 0;
            
            % Take the FFT of the rxDishReturns matrix and the replica
            numCols = size(rxDishReturns, 1);
            
            % Compute the Range Responses
            X = fft(rxDishReturns, numCols, 1);
            H = fft(replica, numCols, 1);
                        
            Y = X.*H;
            
            y = ifft(Y, numCols*o.fastTimePaddingFactor, 1);
            
            % Compensate for the Matched Filter delay
            matchingdelay = o.fastTimePaddingFactor*size(replica,1)-1;
            y = buffer(y(matchingdelay+1:end),size(y,1));
            y = sum(y, 2); % Coherently integrate the pulses together
            
            %% Create the axes for Range
            fast_time_grid = unigrid(0,...
                1/(o.fs*o.fastTimePaddingFactor),1/o.PRF,'[)');
            
            prop_speed = physconst('LightSpeed');
            range_gates = prop_speed*fast_time_grid/2;
            
            NoiseBandwidth = o.fs;
            noisepow = physconst('Boltzmann')*...
                systemp(o.noiseFig, o.refTemp) * NoiseBandwidth;
            mfgain = replica(:,1)'*replica(:,1);
            noisepower_proc = mfgain*noisepow;
            y = pow2db(abs(y).^2/noisepower_proc);
            % Apply a range offset proportional to account for the radar
            % unambiguous range
            o.rangeOffset = max(range_gates) * (o.numCalls - 1);
            o.timeOffset = max(range_gates)/physconst('lightspeed') * ...
                (o.numCalls - 1);
            
            o.rxPacketTimes = linspace(0, 1/o.PRF, ...
                size(rxDishReturns,2)) + o.rangeOffset;
            
            figure();
            plot((o.rangeOffset + ...
                range_gates)/1e3, y)
            
            xlabel('Range (km)')
            ylabel('Log Magnitude (dB)')
            %title('Range Cut of Range-Doppler Response Surface')
            if o.deltaF <= 0
                titleStr = ['Range-Doppler Range Response ' ...
                    'f_{Sweep} = ' num2str(o.fSweep/1e6) ' MHz, PRF = ' num2str(o.PRF) ', tgtPos = ' ...
                    num2str(ceil(norm(o.tgtPos)/1e3)) ' km, numPulses = ' num2str(size(rxDishReturns,2))];
            else
                titleStr = ['Synthetic Wideband Range Response ' ...
                    'f_{Sweep} = ' num2str(o.fSweep/1e6) ' MHz, PRF = ' num2str(o.PRF) ', tgtPos = ' ...
                    num2str(ceil(norm(o.tgtPos)/1e3)) ' km, numPulses = ' num2str(size(rxDishReturns,2)), ...
                    ' \Delta_F = ' num2str(o.deltaF/1e3) ' KHz'];
            end
            title(titleStr)            
            grid minor
              
            
        end
    end
end

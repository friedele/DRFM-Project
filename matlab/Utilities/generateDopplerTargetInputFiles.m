%% Get Radar parameters
path = 'C:\Users\friedele\Repos\DRFM';
radarFile = fullfile(path,'inputs','radar.xlsx');
[~,radar] = getParameters(radarFile,"radar");

% Define structures for processing
target = struct;
target.pos = 0;
target.rcs = 10;
target.type = 0;  % 0 - False, 1 - Real Target
target.pw = 0;
target.vel = 0;
target.range = 0;
target.pwEnum = 0;
target.idx = 0;
target.prf = [25 50 100 200 400 800 1600 3200];
target.header = ['Target_Idx,' 'Target_Range_m,' 'Target_Vel_m,' 'Target_RCS,' 'Target_Type,' 'pw,' 'Delta_Range_m'];
headerFields = split(target.header,',');

% Write the initial header to the Target file
path = 'C:\Users\friedele\Repos\DRFM\inputs';
targetFile = fullfile(path,'orderedTargets/','dopplerMaskTargets.xlsx');
writecell(headerFields',targetFile);

%% Get Target Inputs
% Orderly run through all the PRFs and ranges while incrementing through
% the targets

% Variables

% Pick some arbitutaury values
% target.range = randi([minRange maxRange],1);
% target.vel = randi([minVel maxVel],1);
% minVel = randi([minVel maxVel],1);
% maxVel = minVel+radar.deltaVel(idx);

prf = target.prf(1);
target.range = radar.maxTgtRng(1);
radarConfigIdx = 1;
totalTargets = 1;
target.idx = 1;

for numTargets=3:16
    while (radar.minTgtRng(size(target.prf ,2))<=target.range)
        minRange = int32(radar.minTgtRng(radarConfigIdx));
        maxRange = int32(radar.maxTgtRng(radarConfigIdx));
        target.pwEnum = radar.pwEnum(radarConfigIdx);
        target.type = 0;

        if (maxRange==3000e3 || maxRange==2000e3 || maxRange==1000e3)
            target.range=target.range-50e3;
            deltaRange = 50e3;
        elseif (maxRange==550e3 || maxRange==270e3)
            target.range=target.range-50e3;
              deltaRange = 50e3;
        elseif(maxRange==135e3)
            target.range=target.range-4e3;
              deltaRange = 4e3;
        else
            target.range=target.range-0.75e3;
              deltaRange = 0.75e3;
        end
   
        % checkRange and switch pw
        radarConfigIdx = getPulsewidth(radarConfigIdx,radar,target.range);
        minVel = radar.minVel(radarConfigIdx);
        maxVel = radar.maxVel(radarConfigIdx);
        target.pwEnum = radarConfigIdx;
        targetVec = [target.idx, target.range, target.vel, target.rcs, target.type, target.pwEnum, deltaRange];
        target.idx = target.idx+1;
        
        % Group by number of targets
        if (target.idx==numTargets+1)
            target.idx=1;
            target.vel = randi([minVel maxVel],1);
        end
        totalTargets = totalTargets+1;

        if (target.range >= minRange)
            writematrix(targetVec,targetFile,'WriteMode','append')
        end
    end
    target.range = radar.maxTgtRng(1); % Reset target range and pw enum for the next loop
    radarConfigIdx = 1;

end
  fprintf('Total Targets: %d\n',totalTargets)


  function pwEnum = getPulsewidth(idx,radarParams,range)
    targetRangeSwath = [radarParams.minTgtRng(idx), radarParams.maxTgtRng(idx)];
    if (targetRangeSwath(1) <= range) && (range <= targetRangeSwath(2))
        idx = idx;
    else
        idx = idx+1;
    end

    if (idx>8)
        pwEnum = 8;
    else
        pwEnum = idx;
    end

end
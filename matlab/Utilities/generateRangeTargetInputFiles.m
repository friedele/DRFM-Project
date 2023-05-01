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
targetFile = fullfile(path,'orderedTargets/','rangeMaskTargets.xlsx');
writecell(headerFields',targetFile);

%% Get Target Inputs
% Orderly run through all the PRFs and ranges while incrementing through
% the targets

prf = target.prf(1);
target.range = radar.maxTgtRng(1);
radarConfigIdx = 1;
totalTargets = 1;
target.idx = 1;

for numTargets=3:16
    while (radar.minVel(size(target.prf ,2))<=target.vel)
        minVel = int32(radar.minVel(radarConfigIdx));
        maxVel = int32(radar.maxVel(radarConfigIdx));
        target.pwEnum = radar.pwEnum(radarConfigIdx);
        target.type = 0;

        if (maxVel==250 || maxVel==350 || maxVel==500)
            target.vel=target.vel-5;
            deltaVel = 15;
        else
            target.vel=target.vel-10;
            deltaVel = 100;
        end
   
        % checkRange and switch pw
        radarConfigIdx = getPulsewidth(radarConfigIdx,radar,target.vel);
        minRange = radar.minTgtRng(radarConfigIdx);
        maxRange = radar.maxTgtRng(radarConfigIdx);
        target.pwEnum = radarConfigIdx;
        targetVec = [target.idx, target.range, target.vel, target.rcs, target.type, target.pwEnum, deltaVel];
        target.idx = target.idx+1;
        
        % Group by number of targets
        if (target.idx==numTargets+1)
            target.idx=1;
            target.range = randi([minRange maxRange],1);
        end
        totalTargets = totalTargets+1;

        if (target.vel >= minVel)
            writematrix(targetVec,targetFile,'WriteMode','append')
        end
    end
    target.vel = radar.maxVel(1); % Reset target range and pw enum for the next loop
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
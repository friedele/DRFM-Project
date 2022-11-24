%% Generate targets
function generateTargets(targetFile,type,numTargets,prf)

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
target.num = numTargets;
target.pw = 0;
target.idx = 0;
target.prf = prf;
target.header = ['Target_Idx,' 'Target_Range_m,' 'Target_Vel_m,' 'Target_RCS,', 'Target_Type,' 'pw'];
headerFields = split(target.header,',');

% Variables
idx = find(radar.prf==target.prf);
minRange = radar.minTgtRng(idx);
maxRange = radar.maxTgtRng(idx);
minVel = radar.minVel(idx);
maxVel = radar.maxVel(idx);
target.pw = radar.pwEnum(idx);

writecell(headerFields',targetFile);

% Pick some arbitutaury values
target.range = randi([minRange maxRange],1);
target.vel = randi([minVel maxVel],1);
minVel = randi([minVel maxVel],1);
maxVel = minVel+radar.deltaVel(idx);

% Generate target file
for n=1:target.num
    switch type
        case ('random')
            target.range = randi([minRange maxRange],1);
            target.vel = randi([minVel maxVel],1);
        case ('rangemask')
            target.vel = randi([minVel maxVel],1);
        case('dopplermask')
            target.range = randi([minRange maxRange],1);
        case ('real')
        case('combined')
            if(n>target.num/2)
                target.range = randi([minRange maxRange],1);
            else
                target.vel = randi([minVel maxVel],1);
            end
        otherwise
            disp('Choices are random, rangemask, dopplermask, combined', 'real')
            return
    end
    target.idx = n;
    targetVec = [n, target.range, target.vel, target.rcs, target.type, target.pw];
    writematrix(targetVec,targetFile,'WriteMode','append')
end % End for Loop

end

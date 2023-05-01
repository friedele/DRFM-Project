function [opts,M] = getParameters(filename,type)
% DESCRIPTION: This function defines the parameters for the phasedArrayRadar
% simulation.  There are two types of structures defined which are for the
% target and the radar.
% INPUTS:  filename:  string  (Excel spreadsheet file)
%          type:  string  (types - 'radar' or 'targets')
% EXAMPLE: [~,tgt] = getParameters(tgtFile,"targets");
%          [~,radar] = getParameters(radarFile,"radar");
% See Also:  phasedArrayRadar.m

opts = detectImportOptions(filename);
preview(filename,opts);
M = readmatrix(filename,opts);

switch type
    case "targets"
        target = struct;
        target.id = M(:,1);
        target.pos = M(:,2);
        target.vel = M(:,3);
        target.rcs = M(:,4);
        target.type = M(:,5);
        target.pw = M(:,6);
%        target.deltaRange = M(:,7);
        M = target;
    case "radar"
        radar = struct;
        radar.pwEnum = M(:,1);
        radar.pw = M(:,2);
        radar.duty = M(:,3);
        radar.fc = M(:,4);
        radar.pri = M(:,5);
        radar.prf = M(:,6);
        radar.fs = M(:,7);
        radar.wb = M(:,8);
        radar.nPulses = M(:,9);
        radar.nDoppler = M(:,10);
        radar.peakPwr = M(:,11);
        radar.txGain = M(:,12);
        radar.rxGain = M(:,13);
        radar.noiseFigure = M(:,14);
        radar.minTgtRng = M(:,15);
        radar.maxTgtRng = M(:,16);
        radar.minVel = M(:,17);
        radar.maxVel = M(:,18);
        radar.deltaVel = M(:,19);
        radar.deltaRange = M(:,20);
        M = radar;
    otherwise
        disp("Invalid type use 'radar' or 'targets'.")
end
end
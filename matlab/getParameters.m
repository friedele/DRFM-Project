function [opts,M] = getParameters(filename,type)

opts = detectImportOptions(filename);
preview(filename,opts)
M = readmatrix(filename,opts);

switch type
    case "targets"
        target = struct;
        target.pos = M(:,2);
        target.vel = M(:,3);
        target.rcs = M(:,4);
        M = target;
    case "radar"
        radar = struct;
        radar.pw = M(:,1);
        radar.duty = M(:,2);
        radar.fc = M(:,3);
        radar.pri = M(:,4);
        radar.prf = M(:,5);
        radar.fs = M(:,6);
        radar.wb = M(:,7);
        radar.nPulses = M(:,8);
        radar.nDoppler = M(:,9);
        radar.peakPwr = M(:,10);
        radar.txGain = M(:,11);
        radar.rxGain = M(:,12);
        M = radar;
    case "drfm"
end

end
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
        M = radar;
    case "drfm"
end

end
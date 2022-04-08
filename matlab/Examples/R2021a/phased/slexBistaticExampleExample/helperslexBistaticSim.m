function helperslexBistaticSim(command)
% This function helperslexBistaticSim is only in support of
% slexBistaticExample. It may be removed in a future release.

%   Copyright 2014 The MathWorks, Inc.

switch command
    case 'openModel'
        open_system('slexBistaticExample');
        helperslexBistaticSim('closePlots');
    case 'closePlots'
        close_system('slexBistaticExample/Visualization/Range Doppler Map');
    case 'runModel'
        sim('slexBistaticExample');
        helperslexBistaticSim('closePlots');
    case 'showRangeDopplerMap'
        openForPublish('slexBistaticExample/Visualization/Range Doppler Map');
    case 'showTransmitter'
        openForPublish('slexBistaticExample/Radar Transmitter');
    case 'showTarget'
        openForPublish('slexBistaticExample/Target 1');
    case 'showReceiver'
        openForPublish('slexBistaticExample/Radar Receiver');
    case 'showRangeDopplerProcessor'
        openForPublish('slexBistaticExample/Range Doppler Processor');
        openForPublish('slexBistaticExample/Range Doppler Processor/Range Doppler Response');
    case 'showRangeDopplerVisual'
        openForPublish('slexBistaticExample/Visualization/Range Doppler Map');
    case 'closeModel'
        close_system('slexBistaticExample',0);
end

function openForPublish(blk)
open_system(blk,'force');

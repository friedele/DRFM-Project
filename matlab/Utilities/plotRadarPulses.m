function plotRadarPulses(datacube,threshold,t,tp,numPulses)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
thresh = sqrt(threshold).*ones(numel(t),1);
for m = 1:numPulses
    subplot(numPulses,1,m);
    tnow = t+tp(m);
    rpulse = abs(datacube(:,m));
    rpulse(rpulse == 0) = eps;   % avoid log of 0
    plot(tnow,pow2db(rpulse.^2),tnow,pow2db(thresh.^2),'r--');

    xlabel('Time (s)');
    ylabel('Power (dBw)');
    axis tight;
    ax = axis;
    ax(4) = ax(4)+0.05*abs(ax(4));
    axis(ax);
    grid on;
    if numPulses > 1
        title(sprintf('Pulse %d',m));
    end
end
%% Simulating a Bistatic Radar with Two Targets
% This example shows how to simulates a bistatic radar system with two
% targets. The transmitter and the receiver of a bistatic radar are not
% co-located and move along different paths. 

%   Copyright 2014 The MathWorks, Inc.

%% Exploring the Example
% The following model shows an end-to-end simulation of a bistatic radar
% system. The system is divided into three parts: the transmitter
% subsystem, the receiver subsystem, and the targets and their propagation
% channels. The model shows the signal flowing from the transmitter,
% through the channels to the targets and reflected back to the receiver.
% Range-Doppler processing is then performed at the receiver to generate
% the range-Doppler map of the received echoes.

helperslexBistaticSim('openModel');

%%
% *Transmitter* 
% 
% * |Linear FM| - Creates linear FM pulse as the transmitter waveform. The
% signal sweeps a 3 MHz bandwidth, corresponding to a 50-meter range
% resolution.
% * |Radar Transmitter| - Amplifies the pulse and simulates the transmitter
% motion. In this case, the transmitter is mounted on a stationary platform
% located at the origin. The operating frequency of the transmitter is 300
% MHz.

helperslexBistaticSim('showTransmitter');

%% 
% *Targets*
%
% This example includes two targets with similar configurations. The
% targets are mounted on the moving platforms.
% 
% * |Tx to Targets Channel| - Propagates signal from the transmitter to the
% targets. The signal inputs and outputs of the channel block have two
% columns, one column for the propagation path to each target.
% * |Targets to Rx Channel| - Propagates signal from the targets to the
% receiver. The signal inputs and outputs of the channel block have two
% columns, one column for the propagation path from each target.
% * |Targets| - Reflects the incident signal and simulates both targets
% motion. This first target with an RCS of 2.5 square meters is
% approximately 15 km from the transmitter and is moving at a speed of 141
% m/s. The second target with an RCS of 4 square meters is approximately 35
% km from the transmitter and is moving at a speed of 168 m/s. The RCS of
% both targets are specified as a vector of two elements in the Mean radar
% cross section parameter of the underlying Target block.
%
%

helperslexBistaticSim('showTarget');

%%
% *Receiver*
%
% * |Radar Receiver| - Receives the target echo, adds receiver noise, and
% simulates the receiver motion. The distance between the transmitter and
% the receiver is 20 km, and the receiver is moving at a speed of 20 m/s.
% The distance between the receiver and the two targets are approximately 5
% km and 15 km, respectively.

helperslexBistaticSim('showReceiver');

%%
% * |Range-Doppler Processing| - Computes the range-Doppler map of the
% received signal. The received signal is buffered to form a 64-pulse burst
% which is then passed to a range-Doppler processor. The processor performs
% a matched filter operation along the range dimension and an FFT along the
% Doppler dimension. 

helperslexBistaticSim('showRangeDopplerProcessor');

%% Exploring the Model
% Several dialog parameters of the model are calculated by the helper
% function <matlab:edit('helperslexBistaticParam')
% helperslexBistaticParam>. To open the function from the model, click on
% |Modify Simulation Parameters| block. This function is executed once when
% the model is loaded. It exports to the workspace a structure whose fields
% are referenced by the dialogs. To modify any parameters, either change
% the values in the structure at the command prompt or edit the helper
% function and rerun it to update the parameter structure.

%% Results and Displays
% The figure below shows the two targets in the range-Doppler map.

helperslexBistaticSim('runModel');

%%
% Because this is a bistatic radar, the range-Doppler map above actually
% shows the target range as the arithmetic mean of the distances from the
% transmitter to the target and from the target to the receiver. Therefore,
% the expected range of the first target is approximately 10 km, ((15+5)/2)
% and for second target approximately 25 km, ((35+15)/2). The range-Doppler
% map whos these two values as the measured values.
%
% Similarly, the Doppler shift of a target in a bistatic configuration is
% the sum of the target's Doppler shifts relative to the transmitter and
% the receiver. The relative speeds to the transmitter are -106.4 m/s for
% the first target and 161.3 m/s for the second target while the relative
% speeds to the receiver are 99.7 m/s for the first target and 158.6 m/s
% for second target. Thus, the range-Doppler map shows the overall relative
% speeds as -6.7 m/s (-24 km/h) and 319.9 m/s (1152 km/h) for the first
% target and the second target, respectively, which agree with the expected
% sum values.

%% Summary
% This example shows an end-to-end bistatic radar system simulation with
% two targets. It explains how to analyze the target return by plotting a
% range-Doppler map.

helperslexBistaticSim('closeModel')

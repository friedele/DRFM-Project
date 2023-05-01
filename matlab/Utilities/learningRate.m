% A = [0,1,2;3,4,5;6,7,8];
% B = [5,8,1;8,0,1;6,4,1];
% C = [6,3,4,4,5,0,3;4,7,4,0,4,0,4;7,0,2,3,4,5,2;3,7,5,0,3,0,7;5,8,1,2,5,4,2;8,0,1,0,6,0,0;6,4,1,3,0,4,5];
% D = [3,0,1,2,7,4;1,5,8,9,3,1;2,7,2,5,1,3;0,1,3,1,7,8;4,2,1,6,2,8;2,4,5,2,3,9];
% E = [3,0,1;1,5,8;2,7,2];
% F = [0,0,0,1,1,1;0,0,0,1,1,1;0,0,0,1,1,1;0,0,0,1,1,1;0,0,0,1,1,1;0,0,0,1,1,1];
% filter1 = [0,1;2,3];
% filter2 = [1,0,-1;1,0,-1;1,0,-1];
% filter3 = [1,0,-1;2,0,-2;1,0,-1];
% xcorr(A,filter1) % Looking for a 5x5 output (7-3+1) for all possible spatial alignments
% convn(D,filter2)
% conv2(E,filter2)
% conv2(filter3,F)
% This code provides values of a, A, and c in the gains a_k=a/(k+1+A)^alpha and 
% c_k = c/(k+1)^.101.  Code assumes Bernoulli +/- 1 distribution for the delta elements. 
%
% Note: if get some error "Subscript indices must be integer values"
% then use "clear" function and redo.
%
% Specify the dimension, loss function, and i.c. for theta; also alpha for gain 
% selection (alpha = 0.602 is often used as a companion to gamma = 0.101). 
%
global sigma p z	% sigma is noise level used in loss calls; z is global 
						% random variable for use in generating noise in loss
						% function calls(so that seed can be set here).
sigma=2;
p=1;
loss='loss_TBD';
theta=0.01;
alpha=0.602;
%
% User input on measurement noise level, expected no. of iterations in the SPSA run,
% desired magnitude of change in the theta elements, the number of SPSA gradient approximations
% that will be averaged, and the no. of loss evaluations to be used in the gain calculations
% here (note this no. should be divisible by twice the no. of averaged gradients).
%
step= input('What is the initial desired magnitude of change in the theta elements? ');
gavg= input('How many averaged SP gradients will be used per iteration? ');
A = .10*input('What is the expected number of loss evaluations per run? ')...
   /(2*gavg);
c = input('What is the standard deviation of the measurement noise at i.c.? ');
c = max(c/gavg^0.5, .0001); %averaging set up for independent noise
NL = input('How many loss function evaluations do you want to use in this gain calculation? ');
%
% Do the NL/(2*gavg) SP gradient estimates
%
rand('seed',31415927)	% Seed for delta generation
randn('seed',111113); 	% Seed for noise in loss measurements
gbar=zeros(p,1);
for i=1:NL/(2*gavg)
   ghat=zeros(p,1);
   for j=1:gavg
      delta=2*round(rand(p,1))-1;
      thetaplus=theta+c*delta;
      thetaminus=theta-c*delta;
      yplus=feval(loss,thetaplus);
      yminus=feval(loss,thetaminus);
      ghat=(yplus-yminus)./(2*c*delta)+ghat;
   end
   gbar=gbar+abs(ghat/gavg);
end
gbar
meangbar=mean(gbar);
meangbar=meangbar/(NL/(2*gavg));
a=step*((A+1)^alpha)/meangbar;
%
% Print out of c, A, and a
%
c
A
a

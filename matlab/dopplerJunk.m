freq = 9.5e9;
v = 150;
lambda = physconst('LightSpeed')/freq;
dopplershift = speed2dop(v,lambda)


lambda = physconst('LightSpeed')/freq;
speed = dop2speed(dopplershift,lambda)
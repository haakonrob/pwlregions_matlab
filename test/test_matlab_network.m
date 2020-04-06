addpath('/home/haakon/Projects/affine-compositions/workspace/matlab/')
% addpath('..')
rng(15500);

% Start up parallel pool
pool = gcp;


%% Nonlinear pendulum model
g = 9.81;
L = 5;
m = 1;
d = 0.1;
Ts = 0.1;
xmax =  [pi,8];
xmin = -[pi,8];

y_dot = @(t, x) [x(2,:) ; -g/L * sin(x(1,:)) - d/m * x(2,:)];
y_dot_u = @(t, x, u) [x(2,:) ; -g/L * sin(x(1,:)) - d/m * x(2,:) + u/m];


%% Generate pendulum data
% State and input size
nx = 2;
nu = 0;

% Sample our nonlinear model
theta_dist = makedist('uniform', 'lower', -3*pi, 'upper', 3*pi);
dtheta_dist = makedist('normal', 'mu', 0, 'sigma', 15);

x = [random(theta_dist, [1,10000]); random(dtheta_dist, [1,10000])];
y = y_dot(0, x);


%% Define and Train the network
layers = [
    imageInputLayer([nx+nu,1,1],"Name","sequence","Normalization","none")
    fullyConnectedLayer(15,"Name","fc1")
    reluLayer
    fullyConnectedLayer(5,"Name","fc2")
    reluLayer
    fullyConnectedLayer(2, "Name", 'output', 'BiasLearnRateFactor', 0)
    regressionLayer("Name","regressionoutput")
];


%% Train the network (not too long, we just want to test the pwl code)
% SGDM didn't work at all on the workstation, adam seems to give decent
% results
options = trainingOptions('adam', 'initialLearnRate', 0.001, 'MaxEpochs', 3, 'ExecutionEnvironment', 'cpu');
[net, info] = trainNetwork(reshape(x,[nx+nu,1,1,length(x)]), y', layers, options); 


%% Convert network to PWL form
% Get linear regions of network within a box of side length π
regs = pwl_matlab(net, makebox(nx+nu, pi), 'verbose', true);
f = figure(1);
regs.plot;
axis equal
title("Linear regions of network trained on pendulum data", 'interpreter', 'latex');
xlabel("Angle $\theta$ (rad)", 'interpreter', 'latex');
ylabel("Angular Velocity $\dot{\theta}$ ($ms^{-1}$)", 'interpreter', 'latex')
zlabel("Input torque $u$ (N)")


%% Check network output

% Add functions for each separate input
for i = 1:length(regs)
P = regs(i).Data.P;
regs(i).addFunction(AffFunction(P(1,1:end-1), P(1, end)), 'f1');
regs(i).addFunction(AffFunction(P(2,1:end-1), P(2, end)), 'f2');
end

f = figure(2);
regs.fplot('f1');
axis equal
title("First output of network($\dot{\theta}$)", 'interpreter', 'latex');
xlabel("$\theta$ [rad]", 'interpreter', 'latex');
ylabel("$\dot{\theta}$ [rad $s^{-1}$]", 'interpreter', 'latex');
zlabel("Network output $\dot{\theta}$ [rad$s^{-1}$]", 'interpreter', 'latex');

f = figure(3);
regs.fplot('f2');
axis equal
title("Second output of network($\ddot{\theta}$)", 'interpreter', 'latex');
xlabel("$\theta$ [rad]", 'interpreter', 'latex');
ylabel("$\dot{\theta}$ [rad $s^{-1}$]", 'interpreter', 'latex');
zlabel("Network output $\ddot{\theta}$ [rad$s^{-2}$]", 'interpreter', 'latex');


%% Fixed Parameters

% Using an infinitely large box seems to cause numerical issues, which
% results in regions getting dropped.
space = makebox(2,100);

H = [
    0.8731   -1.4007    1.6084;
   -1.1209    0.3543   -0.7947;
   -0.8365    0.6661   -2.0656;
    1.2563    0.3641    1.8610;
   -0.9788    0.2106    0.0458;
    0.3570    1.8528   -0.6504;
    1.4998    0.4818    0.2836;
   -1.3154   -0.7027    0.4706
    1.8627   -0.1659    1.0585;
    0.4716   -0.4408    1.2326;
   -0.7657   -0.8472    0.8856;
   -0.4381   -0.0177   -1.6496;
   -0.8593    1.5485   -1.6471;
    0.4767   -0.6759    0.4563;
    0.3074    0.5772   -0.6269;
    0.7696   -0.2874   -0.0178;
    0.4054    0.8327    0.3680;
    1.1113    0.5368   -0.3091;
    0.2759   -0.0493    0.7686;
   -0.1304    1.0935    1.4115;
         0         0    1.0000
];

%% Random parameters (for random initialisations)

dim = 2;
N = 10;
H = [randn(N,dim+1); [zeros(1,dim),1]];
space = makebox(dim,100);

%% Compare the partitioning functions
[regs1,P1] = partition_regions(space, {H}); 
[regs2,P2] = partition_regions2(space, {H}); 
assert(length(regs1) == length(regs2));

%% If possible, plot
dim = size(H, 2)-1;
if dim <= 3
    restricted_view = makebox(dim, 10);
    figure(1); 
    plot(regs1 & restricted_view, 'alpha', 0.2); 
    axis equal;
    title("Intersection algorithm");

    figure(2); 
    plot(regs2 & restricted_view, 'alpha', 0.2);
    axis equal;
    title("Enumeration algorithm");
end
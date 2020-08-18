test_dir = fileparts(which(mfilename));
addpath(fullfile(test_dir,'../src/hyperplane_arrangements'));

H = [
    0.9302    0.6540   -0.4880;
    2.9320    0.2604   -0.8153;
   -0.0166    1.1946   -0.5222;
   -0.5709    0.8113    0.8727;
   -0.2517   -0.6676    0.7177;
   -1.0494   -1.4847    1.5142;
    1.6845    0.4984    0.2431;
    0.6097    0.4157   -1.4223;
    0.4874    0.4794    0.6939;
   -0.7937   -2.1184   -0.5649;
         0         0    1.0000
];

% Large square of space. Notably, contains all of the vertices of the
% arrangement.
space = Polyhedron('lb', [-100,-100], 'ub', [100,100]);

[regs1,P1] = partition_regions(space, {H}); 
[regs2,P2] = enumerate_regions(space, {H}); 

assert(length(regs1) == 55, "Intersection algorithm gave wrong number of regions: "+ length(regs1));
assert(length(regs2) == 55, "Enumeration algorithm gave wrong number of regions: "+ length(regs1));

% Space is far away from origin, contains none of the intersections of the
% hyperplanes. Does this affect the solution? There should be 10 regions.
space = Polyhedron('lb', [50,-100], 'ub', [100,100]);

[regs1,P1] = partition_regions(space, {H}); 
[regs2,P2] = enumerate_regions(space, {H}); 

assert(length(regs1) == 10, "Intersection algorithm gave wrong number of regions: "+ length(regs1));
assert(length(regs2) == 10, "Enumeration algorithm gave wrong number of regions: "+ length(regs1));

%% Visual comparison
% dim = size(H, 2)-1;
% if dim <= 3
%     figure(1); 
%     plot(regs1, 'alpha', 0.2); 
%     axis equal;
%     title("Intersection algorithm");
% 
%     figure(2); 
%     plot(regs2, 'alpha', 0.2);
%     axis equal;
%     title("Enumeration algorithm");
% end
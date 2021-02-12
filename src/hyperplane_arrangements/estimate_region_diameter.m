function width = estimate_region_diameter(reg)

c = reg.interiorPoint.x;  % Find centre, low cost LP
rays = randn(reg.Dim, 2048); % Generate random rays (TODO, how to choose this wrt dimension?)+1
rays = rays ./ vecnorm(rays); % Normalise so t values represent actual length

W = reg.H(:,1:end-1);
b = reg.H(:,end);
T = (b - W*c)./(W*rays); % Intersect ray with all hyperplanes

TT = T;
TT(T<=0) = inf;
T1 = min(TT,[],1); % Shortest forward rays

TT = T;
TT(T>=0) = -inf;
T2 = max(TT,[],1); % Shortest backward rays
width = max(abs(T1 - T2));


end
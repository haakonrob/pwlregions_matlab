function points = raycast_points(reg, N)

c = reg.interiorPoint.x;  % Find centre, low cost LP

if isempty(c)
    points = [];
    return
end

rays = randn(reg.Dim, N); % Generate random rays (TODO, how to choose this wrt dimension?)+1
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

points = [rays .* T1, rays.*T2] + c;


end
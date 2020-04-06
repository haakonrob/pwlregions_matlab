function box = makebox(n, varargin)
%MAKEBOX Summary of this function goes here
parser = inputParser;
parser.addRequired('n', @(x) isnumeric(x) && mod(x,1)==0);
parser.addOptional('width', inf, @(x) numel(x)==1 || numel(x)==n);
parser.parse(n, varargin{:});
w = parser.Results.width;

% box = Polyhedron([eye(n);-eye(n)], w*ones(2*n,1));
box = Polyhedron('lb', -w.*ones(1,n), 'ub', w.*ones(1,n));
box.minHRep;
box.Data.Pk = eye(n+1);
end


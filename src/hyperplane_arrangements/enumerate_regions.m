function [regs_new, Ps_new] = enumerate_regions(regs, Ps)
%PARTITION Partitions all regions into PWA regions.

assert(length(regs) == length(Ps), "Assertion failed, number of regions and P matrices must be equal")

% Place regions in cell array, record number of subregions found
N = length(regs);
regions_cell = mat2cell(regs, ones(1,N), [1]);  
number_subregions_found = zeros(1,N);

% For every subsequent hyperplane, bisect the currently known regions
% recursively
for r = 1 : N
    % Load working region and corresponding affine transformation
    reg = regions_cell{r};
    P = Ps{r};
    
    subregions = enum_regs(reg, P);
    
    % Filter out empty sets and lower dim intersections
    regions_cell{r} = subregions(~subregions.isEmptySet & subregions.isFullDim);
    
    % Record number of subregions found
    number_subregions_found(r) = length(regions_cell{r});
end

% Create flat array of polyhedra
regs_new = [regions_cell{:}];

% Update Ps by finding out which neurons (rows of P) are inactive in each
% region. The corresponding rows can then be set to zero in the P matrix
% for that region.
Ps_new = cell(1,length(regs_new));
for r = 1:length(number_subregions_found)
    for c = 1:number_subregions_found(r)
        i = sum(number_subregions_found(1:r-1))+c;  % ith region
        P = Ps{r};
        active = max(0, (P * [regs_new(i).interiorPoint.x ; 1])) > 0;
        P(~active, :) = 0; 
        P(end,end)=1;
        Ps_new{i} = P;        
    end
end

end

function regs = enum_regs(initial_reg, H)        
    % TODO Handle case where dim > N
    dim = size(H,2)-1;
    N = size(H,1)-1;  % The last row is expected to be [0, ... 0, 1]
    assert(dim <= N, "dim > N is currently not supported")
    pos_map = containers.Map;
    vertices = cell(nchoosek(N, dim), 1);
    pos = cell(nchoosek(N, dim), 1);
    regs = cell(nchoosek(N, dim), 1);

    comb = [];
    for i = 1:nchoosek(N, dim)
        comb = next_comb(N, dim, comb);
        vertices{i} = linsolve(H(comb, 1:end-1), H(comb, end));
        if ~initial_reg.contains(vertices{i})
           continue; 
        end
        if all(isfinite(vertices{i}))
            p = sign(H * [vertices{i} ; -1]);
            p(comb) = 0;
            % For this vertex, try to find new regions by varying the signs
            % of the intersecting hyperplanes
            [regs{i}, pos_map] = get_adj_regs(H, p, comb, pos_map);
        else
            pos{i} = nan;
        end
    end
    
    regs = [regs{:}] ;
    if isempty(regs)
        regs = initial_reg;
    else
        regs = regs & initial_reg;
    end
end


%% Helper Functions

function [regs, pos_map] = get_adj_regs(H, pos, idxs, pos_map)
    A = H(:,1:end-1);
    B = H(:,end);
    
    regs = cell(2^length(idxs), 1);
    signs = -ones(1, length(idxs));
    for i = 1:2^length(idxs)
        pos(idxs) = signs;
        key = num2str(pos==1)';
        if ~pos_map.isKey(key)
            pos_map(key) = true;
            regs{i} = Polyhedron('H', [A , -B].* pos);
        end
        signs = next_bin(signs);
    end
    regs = [regs{:}];
end

function bin = next_bin(bin)
    % Given a "binary" array such as [1,-1,1,-1], increment the array by
    % one (i.e. [-1,1,1,-1]). If the array is [1,1,1,1], return [].
    
    d = length(bin);
    if all(bin == 1)
        bin = [];
    end
    for i = 1:length(bin)
        if bin(i) == -1
            bin(i) = 1;
            break;
        else
            bin(i) = -1;
        end
    end
end

function comb = next_comb(N, dim, varargin)
    % Used to iterate through combinations.
    if nargin < 3
        comb = 1:dim;   
    else
        comb = varargin{1};
    end
    
    if N < dim
        comb = [];
    elseif N == dim || isempty(comb)
        comb = 1:dim;
    elseif comb == N-dim+1:N
        comb = [];
    else
        for idx = dim:-1:1
            if comb(idx) < N - (dim-idx)
                comb(idx:end) = comb(idx)+1 : comb(idx)+1 + (dim-idx);
                break;
            end
        end
    end
end
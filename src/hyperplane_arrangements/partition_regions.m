function [regs_new, Hs_new] = partition_regions(regs, Hs)
%PARTITION_REGIONS Finds all regions of a hyperpalne arraangement though iterative bisection.

assert(length(regs) == length(Hs), "Assertion failed, number of regions and hyperplane arrangements must be equal (try Hs as a cell array)")
% Divide the regions up enough to parallelise the work effectively.

% Place regions in cell array, record number of subregions found
N = length(regs);
regions_cell = mat2cell(regs, ones(1,N), [1]);  
number_subregions_found = zeros(1,N);

% For every subsequent hyperplane, bisect the currently known regions
% recursively
for r = 1 : N
    % Load working region and corresponding affine transformation
    reg = regions_cell{r};
    H = Hs{r};
    
    % For each hyperplane recursively 
    % partition regions using the hyperplane w'x = -b
    for i = 1:(size(H,1))
        if ~all(H(i,1:end-1) == 0)
            hplane = Polyhedron( 'Ae', H(i,1:end-1), 'Be', -H(i,end));
            rec_regions(hplane, reg);
        end
    end
    
    % Flatten the tree of regions (leaves only) into a flat list
    subregions = polyhedron_dfs(reg);
    
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
Hs_new = cell(1,length(regs_new));
for r = 1:length(number_subregions_found)
    for c = 1:number_subregions_found(r)
        i = sum(number_subregions_found(1:r-1))+c;  % ith region
        H = Hs{r};
        active = max(0, (H * [regs_new(i).interiorPoint.x ; 1])) > 0;
        H(~active, :) = 0;
        H(end,end) = 1;  % Just to be sure
        Hs_new{i} = H;        
    end
end

end

%% Helper functions

% Recurses through the tree of regions, checking for intersections with a
% hyperplane
function rec_regions(hplane, reg)
    if reg.doesIntersect(hplane) % then add two children
        if ~isfield(reg.Data, 'children') || isempty(reg.Data.children)
            new_regs = ...
                [reg & Polyhedron( hplane.Ae, hplane.be) , ...
                 reg & Polyhedron(-hplane.Ae,-hplane.be)];
            % Make sure that only full-dim intersections are passed on
            reg.Data.children = new_regs(new_regs.isFullDim);
%             for j = 1:length(reg.Data.children)
%                 reg.Data.children(j).Data.P = reg.Data.P;
%             end
        else % recurse through the children of reg
            for i = 1:length(reg.Data.children)
                rec_regions(hplane, reg.Data.children(i));
            end
        end
    end
end

% Flattens the tree of regions
function children = polyhedron_dfs(regs)
    children = [];
    for i = 1:length(regs)
        if ~isfield(regs(i).Data, 'children') || isempty(regs(i).Data.children)
            children = [children, regs(i)];
        else
            children = [children, polyhedron_dfs(regs(i).Data.children)];
        end
    end
end



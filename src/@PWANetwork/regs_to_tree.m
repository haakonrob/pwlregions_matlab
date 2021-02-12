function s = regs_to_tree(obj)
    global s

    root = obj.root;
    n = count(root,0);
    s = struct();
    s(n).H = [];  % Preallocate
    s(n).children = [];  
    s(n).parent = [];  
    s(n).P = [];  
    
    dfs(root, [], 1);
end

function acc = count(node, acc)
    if isfield(node.Data, 'children') && ~isempty(node.Data.children)
        acc = count(node.Data.children(1), acc);
        acc = count(node.Data.children(2), acc);
    end
    acc = acc+1;
end

function i = dfs(node, parent, i)
    global s
    current = i;
    s(current).parent = parent;

        
    if isfield(node.Data, 'children') && ~isempty(node.Data.children)
        % The last constraint of the first child is the constraint that
        % needs to be satisfied to get to that child. If it is not
        % satisfied, then you go to the other child.
        h = node.Data.children(1).H(end,:); % Lots of redundancy, we only need the most recent constraint
        s(current).H = h; 
        
        s(current).children(1) = i+1;
        i = dfs(node.Data.children(1), current, i+1);

        s(current).children(2) = i+1;  
        i = dfs(node.Data.children(2), current, i+1);
%     elseif isfield(node.Data, 'truncate') && node.Data.truncate == true
%         % If no children and truncated, then we leave this blank for
%         % later approximation
%         s(current).P = [];
    else
        % If there are no children, then this is a leaf node and will be
        % evaluated.
        s(current).P = node.Data.P(1:end-1,:);
    end



    
    

end
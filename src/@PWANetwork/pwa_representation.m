function [regs, root] = pwa_representation(obj, net, varargin)
    %PWA_REPRESENTATION Converts a MATLAB struct describing a neural
    %network to its PWA
    % represention
    % 
    %   [net] is a struct array with fields:
    % s
    %           type: 'dense' | ... (more to come)
    %     activation: 'relu' | ... (more to come)
    %              W: [n×m double]
    %              b: [n×1 double]
    %
    %   [input_space] is an optional parameter, but using it may increase
    %   performance significantly as it reduces the number of linear regions.
    %   It's also much easier to visualise the regions if they are bounded.
    
    % Add additonal methods that are organised in different folders
%     dir = fileparts(which(mfilename));
%     addpath(fullfile(dir, 'hyperplane_arrangements'));
%     addpath(fullfile(dir, 'utils'));
    
    
    
    % Define inputs
    parser = inputParser;
    parser.addRequired( 'net',                      @(s) isa(net, 'struct'));
    parser.addOptional( 'inputs',           [],     @(x) isempty(x) || isnumeric(x));
    parser.addOptional( 'input_space',      [],     @(x) isempty(x) || isa(x, 'Polyhedron'));
    parser.addOptional( 'max_depth',        inf,    @isnumeric);
    parser.addOptional( 'min_diameter',     0,      @isnumeric);
    parser.addOptional( 'min_datapoints',   0,      @isnumeric);
    parser.addOptional( 'approx_data',      false,  @islogical);
%     parser.addOptional( 'data',             [] ,    @isnumeric );
    parser.addParameter('ignore_errors',    false,  @islogical);
    parser.addParameter('verbose',          false,  @islogical);
    
    % Parse inputs
    parser.parse(net, varargin{:});
    inputs = parser.Results.inputs;
    root = parser.Results.input_space;
    max_depth = parser.Results.max_depth;
    min_diameter = parser.Results.min_diameter;
    min_datapoints = parser.Results.min_datapoints;
%     data = parser.Results.data;
    approx_data = parser.Results.approx_data;
    verbose = parser.Results.verbose;
    
    % Allows us to test the detection of layers without throwing errors.
    if parser.Results.ignore_errors
        report = @(msg) disp("An error occurred: "+msg);
    else
        report = @error;
    end
    
    if ~isempty(inputs)
        % You can choose to only look at a subset of the inputs. This is
        % equivalent to supplying a lower dimensional input space on the
        % axes (i.e., the xy plane).
        assert(min(inputs) > 0 && max(inputs) < size(net(1).W, 2));
        net(1).W = net(1).W(:, inputs);
    end
    
    if isempty(root)
       % Input space is R^n, where n is the input dim of the network.
       root = makebox(size(net(1).W, 2), inf);
    else
        % Input space matches the dimensions of the inputs to consider.
        % This doesn't mean that the object isn't lower dimensional! It
        % just means that it lives in the same space as the network inputs.
        % TODO, this keeps the computations in a high dim space, figure out
        % if you can project the weights of the first layer onto this input
        % space, thereby reducing the computational cost significantly.
        assert(root.Dim == size(net(1).W, 2));
    end
    
    
    % Initialise the regions array with the specified input domain and an
    % identity affine transformation.
    regs = root;
    Ps{1} = eye(root.Dim+1);
    regs.Data = struct(); % In case the region has been used as a root before
    regs.Data.depth = 0;  % Node depth is used for truncation
    
    for i = 1:length(net)
        layer = net(i);
        if verbose
            disp("Parsing layer "+i+" ("+class(layer)+")...");
        end
        switch(layer.type)
            case 'dense'
                T = [layer.W, layer.b ; zeros(1,size(layer.W, 2)), 1];
                for k = 1:length(Ps)
                    Ps{k} = T * Ps{k};
                end
            case 'convolution1'
                % TODO Add support for boundary options and strides. Right
                % now I just do the default with stride 1 and no boundary
                % padding.
                % assert(numel(layer.kernel) == length(layer.kernel))
                
                % Padding with zero makes it ignore the last row of the
                % matrix
                kern = layer.kernel(:);
                for k = 1:length(Ps)
                    % Convolve the columns of the P matrix
                     P = conv2(kern,1,Ps{k}(1:end-1,:),'valid');
                     Ps{k} = [P ; zeros(1,size(P,2)), 1];
                end
            case 'convolution2'
                % TODO
                kern = layer.kernel(:);
            otherwise
                report("Unknown layer type for layer "+i)
        end
        
        switch(layer.activation)
            case 'none'
                % Do nothing
            case 'relu'
                hyperplanes = Ps;
                [regs, Ps] = partition_regions(regs, hyperplanes, max_depth, min_diameter, min_datapoints);
            case 'pwa'
                report("general pwa activation not supported yet")
            case 'maxout'
                report("maxout activation not supported yet")
                
            otherwise
                report("Unknown activation for layer "+i)
        end
    end
    
    % Add the corresponding affine function to each region as the
    % functions f1, f2, f3 ...
    % This allows you to compute the output of the PWA function,
    % and is required for fplot() to work.
    % 
    
    global Xtrain Ytrain
    Ynet = obj.eval(Xtrain);
    
    for j = 1:length(regs)
        if isfield(regs(j).Data, 'truncate') && regs(j).Data.truncate
            
            dim = regs(j).Dim;
            cd = regs(j).Data.contains_data;
            m = size(Ynet,1);
            P = zeros(m, dim+1);

            if ~approx_data 
                % Instead of just passing the region through, approximate the
                % network output by applying least squares to the vertices of
                % the region. This involves finding the convex hull of the
                % region, but shouldn't be a problem when max_depth is low
            
%                 % This seems to give the best results, but minVRep is really, really slow
%                 regs(j).minVRep;
%                 V = regs(j).V';
% 
%                 % This works, but never really gets the outermost points
%                 centre = regs(j).interiorPoint.x;
%                 radius = regs(j).interiorPoint.r;
%                 V = centre + 2*radius*(rand(regs(j).Dim, 100)-0.5);
%                 valid = regs(j).contains(V);
%                 V = V(:,valid);
% 
%                 % Extreme points of the region in some random directions, magic
%                 % number of directions, but seems to be much faster than the
%                 % convex hull thing. End up solving many LPs though.
%                 V = randn(regs(j).Dim ,100);
%                 success = false(size(V));
%                 for i = 1:size(V,2)
%                    p = regs(j).extreme(V(:,i)).x; 
%                    if ~isempty(p)
%                      success(i) = true;
%                      V(:,i) = p;
%                    end
%                 end
%                 V = V(:,success);

                % Collect points by raycasting
%                 c = regs(j).interiorPoint.x;  % Find centre, low cost LP
%                 rays = randn(regs(j).Dim, 2000); % Generate random rays (TODO, how to choose this wrt dimension?)
%                 rays = rays./vecnorm(rays);
%                 W = regs(j).H(:,1:end-1);
%                 b = regs(j).H(:,end);
%                 T = (b - W*c)./(W*rays); % Intersect rays with polyhedron
                diam = estimate_region_diameter(regs(j));
                V = zeros(dim,1000);
                start = 1;
                while true
                    V_ = diam/2*(randn(dim, 1000)) + regs(j).interiorPoint.x;
                    V_ = V_(:, regs(j).contains(V_));
                    if size(V_,2) == 0
                        continue
                    end
                    stop = min(start+size(V_,2), 1000);
                    V(:,start:stop) = V_(:,1:stop-start);
                    if stop >= 1000
                        break
                    end
                end
%                 points1 = rays.*min(T + 1e10*(T<=0),[],1) + c; % Shortest forward rays
%                 points2 = rays.*max(T - 1e10*(T>=0),[],1) + c; % Shortest backward rays
%                 V = [points1 , points2];  % Collect points
                
%                 % Just naively sample the network around the centre of the
%                 % region
%                 c = regs(j).interiorPoint.x;  % Find centre, low cost LP
%                 r = regs(j).interiorPoint.r;  % Find centre, low cost LP
%                 V = 4*r*(rand(regs(j).Dim, 1000)-0.5) + c;

                % Evaluate the samples using the network
                Y = obj.eval(V);

                for i = 1:m
                    P(i,:) = fitlm(V', Y').Coefficients.Estimate;
                end
                P = [p ; zeros(1,size(p,2)-1) , 1];
            else
                for i = 1:m
                    P(i,:) = fitlm(Xtrain(:,cd)', Ynet(i,cd)').Coefficients.Estimate;
                end
                b = P(:,1);
                P(:,1:end-1) = P(:,2:end); P(:,end) = b;
                P = [P; zeros(1,dim) , 1];
            end
        else
            P = Ps{j};
        end

%         P = Ps{j};
        % Save full output as function
        if ~isempty(P)
            regs(j).addFunction(AffFunction(P(1:end-1, 1:end-1), P(1:end-1, end)), "f");
            for i = 1:size(P, 1)-1
                regs(j).addFunction(AffFunction(P(i, 1:end-1), P(i, end)), "f"+i);
            end
        end
        % Save local P matrix for later use
        regs(j).Data.P = P;
    end
    
    end
    
    
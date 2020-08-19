function [regs] = pwa_representation(obj, net, varargin)
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
    parser.addRequired( 'net',                 @(s) isa(net, 'struct'));
    parser.addOptional( 'inputs',  [],    @(x) isa(x, 'Polyhedron'));
    parser.addOptional( 'input_space',  [],    @(x) isa(x, 'Polyhedron'));
    parser.addParameter('ignore_errors',false, @islogical);
    parser.addParameter('verbose',false, @islogical);
    
    % Parse inputs
    parser.parse(net, varargin{:});
    input_space = parser.Results.input_space;
    verbose = parser.Results.verbose;
    
    % Allows us to test the detection of layers without throwing errors.
    if parser.Results.ignore_errors
        report = @(msg) disp("An error occurred: "+msg);
    else
        report = @error;
    end
    
    if isempty(input_space)
       % Input space is R^n, where n is the input dim of the network.
       input_space = makebox(size(net(1).W, 2), inf);
    end
    
    % Initialise the regions array with the specified input domain and an
    % identify affine transformation.
    regs = input_space;
    Ps{1} = eye(input_space.Dim+1);
    
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
            case 'convolution'
                % TODO Add support for boundary options
                % assert(numel(layer.kernel) == length(layer.kernel))
                
                % Padding with zero makes it ignore the last row of the
                % matrix
                kern = [0 ;layer.kernel(:)];
                for k = 1:length(Ps)
                    % Convolve the columns of the P matrix
                    Ps{k} = conv2(kern,1,Ps{k},'valid');
                end
            otherwise
                report("Unknown layer type for layer "+i)
        end
        
        switch(layer.activation)
            case 'relu'
                hyperplanes = Ps;
                [regs, Ps] = partition_regions(regs, hyperplanes);
            case 'pwa'
                report("general pwa activation not supported yet")
            case 'maxout'
                report("maxout activation not supported yet")
                
            otherwise
                report("Unknown activation for layer "+i)
        end
    end
    
    % Add the corresponding affine function to each region as the functions
    % f1, f2 , f3 ...
    for j = 1:length(regs)
        P = Ps{j};
        % Save full output as function
        regs(j).addFunction(AffFunction(P(1:end-1, 1:end-1), P(1:end-1, end)), "f");
        for i = 1:size(P, 1)-1
            regs(j).addFunction(AffFunction(P(i, 1:end-1), P(i, end)), "f"+i);
        end
        % Save local P matrix for later use
        regs(j).Data.P = P;
    end
    
    end
    
    
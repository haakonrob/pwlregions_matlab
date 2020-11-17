classdef PWANetwork < handle
    %PWANetwork Summary of this class goes here
    %   Detailed explanation goes here
    
    properties(SetAccess=private, Transient=true)
		layers  % Struct array of the layers of the network
        regions
    end
    
    methods(Access = public)
        function obj = PWANetwork(varargin)
            %PWANetwork Construct an instance of this class
            
            if nargin > 1
               error("Error: Cannot specify more than one network"); 
            elseif nargin < 1
               error("Error: Need to specify network specification. This can be a JSON file, a .mat file, or a struct array."); 
            end
            
            spec = varargin{1};
            
            if ischar(spec) || isstring(spec)7
                obj.layers = obj.from_file(spec); 
            elseif isstruct(spec)
                obj.layers = obj.validate_layers(spec);
            else
                error("asdfError: Need to specify network specification. This can be a JSON file, a .mat file, or a struct array."); 
            end
                           
        end
        
        function regs = pwa(obj, varargin)
            regs = obj.pwa_representation(obj.layers, varargin{:});
            obj.regions = regs;
        end
        
        function plot_output(obj, i)
            obj.regions.fplot(['f' num2str(i)]);
        end
        
        function layers = from_file(obj,filepath)
            % Checks the file type and passes it on
            
            [~, ~, ext] = fileparts(filepath);
            if strcmp(ext, '.mat')
                layers = obj.load_layers_from_mat_file(filepath);
            else
                layers = obj.load_layers_from_json_file(filepath);
            end
            obj.validate_layers(layers);
            
        end
        
        
        function layers = load_layers_from_mat_file(obj,filepath)
            % This code returns the first struct from the given mat file
            layers = [];
            s = load(filepath);
            fn = fieldnames(s);
            for k=1:numel(fn)
                if( isstruct(s.(fn{k})) )
                    layers = s.(fn{k});
                    break;
                end
            end
            if isempty(layers)
                error("Did not find suitable struct array in given mat file")
            end
        end
        
        function layers = load_layers_from_json_file(obj, filepath)
            % Parse the given json file and return the layers
            layers = jsondecode(fileread(filepath));
        end
        
        function to_json(obj, filepath)
            % Output a json file containing the layers of the network
            data = jsonencode(obj.layers);
            fid=fopen(filepath,'w');
            fprintf(fid, data);
        end
        
        function layers = validate_layers(obj,layers)
            if ~isstruct(layers)
                error("Error: layer spec must be struct array")
            elseif ~(length(layers) == numel(layers))
                error("Error: layer spec must be 1D struct array")
            end
        end
        
        function y = eval(obj,x)
            %eval Compute the output of the network given x
            for i = 1:length(obj.layers)
               switch (obj.layers(i).type)
                   case 'dense'
                       x = obj.layers(i).W * x + obj.layers(i).b;
                   otherwise
                       error('Unsupported layer type');
               end
               if obj.layers(i).activation == 'relu'
                    x = max(0,x);
               end
            end
            y = x;
        end
    end
end


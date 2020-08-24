classdef RandomPWANetwork < PWANetwork
    %PWANetwork Summary of this class goes here
    %   Detailed explanation goes here
    
    
    methods(Access = public)
        function obj = RandomPWANetwork(n_inputs, hidden_neurons, varargin)
            %RandomPWANetwork Construct an instance of this class
            neurons = [n_inputs ; hidden_neurons(:)];

            for i = 1:length(neurons)-1
                random_layers(i) = struct(               ...
                    'type', 'dense',                     ...
                    'activation', 'relu',                ...
                    'W', randn(neurons(i+1),neurons(i)), ...
                    'b', randn(neurons(i+1),1)           ...
                );
            end
            % random_layers(2) = struct('type', 'dense', 'activation', 'relu', 'W', randn(neurons(2),neurons(2)), 'b', randn(neurons(2),1));
            % random_layers(3) = struct('type', 'dense', 'activation', 'relu', 'W', randn(neurons(3),neurons(3)), 'b', randn(neurons(3),1));
            obj = obj@PWANetwork(random_layers);                           
        end
    end
end


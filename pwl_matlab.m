function [regs] = pwl_matlab(net, varargin)
%PWL_MATLAB Converts a MATLAB SeriesNetwork into a PWA function.
%
%   input_space is an optional parameter, but using it may increase
%   performance significantly as it reduces the number of linear regions.
%   It's also much easier to visualise the regions if they are bounded. 


% Define inputs
parser = inputParser;
parser.addRequired( 'net',                 @(s) isa(net, 'SeriesNetwork'));
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

% Initialise the regions array with the specified input domain and an
% identify affine transformation.
regs = input_space;
Ps{1} = eye(input_space.Dim+1);

for i = 1:length(net.Layers)
    layer = net.Layers(i);
    if verbose
        disp("Parsing layer "+i+" ("+class(layer)+")...");
    end
    switch(class(layer))
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Input Layers
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'nnet.cnn.layer.SequenceInputLayer'
            idim = net.Layers(i+1).InputSize;
            if (i ~= 1)
                % The input layer must be the first layer, otherwise
                % strange things might happen.
                report("Unexpected input layer "+i+" (SequenceInputLayer), already have input.");
            elseif isempty(input_space)
                % No input space has been specified, create new unbounded 
                % input space
                input_space = makebox(idim, inf);
                regs = input_space;
            elseif idim ~= input_space.Dim
                % The network input is incompatible with the given input
                % space.
                report("Input space dim ("+input_space.Dim+ ...
                      ") does not match input layer dim ("+idim+").");
            end
            
        case 'nnet.cnn.layer.ImageInputLayer'      %Image input layer
            % TODO this layer includes a normalisation layer by default! 
            % Until this is resolved, networks should be defined
            % without normalization: ("normalization", "none")
            idim = net.Layers(i+1).InputSize;
            if (i ~= 1)
                % The input layer must be the first layer, otherwise
                % unspecified things might happen.
                report("Unexpected input layer "+i+" (ImageInputLayer), already have input.");
            elseif isempty(input_space)
                % No input space has been specified, create new unbounded 
                % input space
                input_space = makebox(idim, inf);
                regs = input_space;
            elseif idim ~= input_space.Dim
                % The network input is incompatible with the given input
                % space.
                report("Input space dim ("+input_space.Dim+ ...
                       ") does not match input layer dim ("+idim+").");
            end
        
        case 'nnet.cnn.layer.Image3DInputLayer'      %Image 3D input layer
            idim = net.Layers(i+1).InputSize;
            if (i ~= 1)
                % The input layer must be the first layer, otherwise
                % unspecified things might happen.
                report("Unexpected input layer "+i+" (Image3DInputLayer), already have input.");
            elseif isempty(input_space)
                % No input space has been specified, create new unbounded 
                % input space
                input_space = makebox(idim, inf);
                regs = input_space;
            elseif idim ~= input_space.Dim
                % The network input is incompatible with the given input
                % space.
                report("Input space dim ("+input_space.Dim+ ...
                       ") does not match input layer dim ("+idim+").");
            end
            
        
            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Convolutional and fully connected layers
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'nnet.cnn.layer.FullyConnectedLayer' % Fully connected layer
            W = double(layer.Weights);          
            B = double(layer.Bias);
            T = [W, B ; zeros(1,size(W, 2)), 1];
            for k = 1:length(Ps)
                Ps{k} = T * Ps{k};
            end
            
        case 'nnet.cnn.layer.Image3dInputLayer'      %3-D image input layer
            report("Layer "+i+" (Image3dInputLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.Convolution2dLayer'      %2-D convolutional layer
            report("Layer "+i+" (Convolution2dLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.Convolution3dLayer'      %3-D convolutional layer
            report("Layer "+i+" (Convolution3dLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.GroupedConvolution2dLayer'      %2-D grouped convolutional layer
            report("Layer "+i+" (GroupedConvolution2dLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.TransposedConv2dLayer'      %	Transposed 2-D convolution layer
            report("Layer "+i+" (TransposedConv2dLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.TransposedConv3dLayer'      %Transposed 3-D convolution layer
            report("Layer "+i+" (TransposedConv3dLayer) is not supported yet!");
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Recurrent layers
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'nnet.cnn.layer.LSTMLayer'	%Long short-term memory (LSTM) layer
            report("Layer "+i+" (LSTMLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.BiLSTMLayer'	%Bidirectional long short-term memory (BiLSTM) layer
            report("Layer "+i+" (BiLSTMLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.SequenceFoldingLayer'	%Sequence folding layer
            report("Layer "+i+" (SequenceFoldingLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.SequenceUnfoldingLayer'	%Sequence unfolding layer
            report("Layer "+i+" (SequenceUnfoldingLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.FlattenLayer'	%Flatten layer
            report("Layer "+i+" (FlattenLayer) is not supported yet!");
            
            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Activations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'nnet.cnn.layer.ReLULayer'      %Rectified Linear Unit (ReLU) layer
            [regs, Ps] = partition_regions(regs, Ps);
            
        case 'nnet.cnn.layer.LeakyReLULayer'      %Leaky Rectified Linear Unit (ReLU) layer
            report("Layer "+i+" (LeakyReluLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.ClippedReLULayer'      %Clipped Rectified Linear Unit (ReLU) layer
            report("Layer "+i+" (ClippedReluLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.ELULayer'      %Exponential linear unit (ELU) layer
            report("Layer "+i+" (EluLayer) is not supported because it is not a piecewise linear activation.");
            
        case 'nnet.cnn.layer.TanhLayer'      %Hyperbolic tangent (tanh) layer
            report("Layer "+i+" TanhLayer) is not supported because it is not a piecewise linear activation.");
            
            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Normalisation layers
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'nnet.cnn.layer.BatchNormalizationLayer'
            % BATCH NORMALISATION LAYER This layer recenters and scales
            % its inputs based on the mean and variance it observed during
            % training. This preserves the usefulness of signals within the
            % network, ensuring that one output does not completely
            % dominate the others.
            
            % TODO Check that the output of the layer is the same, not sure
            % if scale or offset is applied to x first.
            scale = double(layer.Scale(:));
            offset = double(layer.Offset(:));
            T = [eye(length(scale)) , offset ; zeros(1,length(scale)), 1];
            Ps = compose_layer(Ps, T);
            
        case 'nnet.cnn.layer.CrossChannelNormalizationLayer'      %	Channel-wise local response normalization layer
            report("Layer "+i+" (CrossChannelNormalizationLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.DropoutLayer'      %	Dropout layer
            % TODO: Dropout is a layer that randomly sets some of its outputs to
            % zero (during training) to help increase the robustness of a 
            % learned model. It is turned off after training, but it might
            % be the case that it scales down its input.
            report("Layer "+i+" (DropoutLayer) is not supported yet!");
            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Pooling layers
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % TODO There are quite a few papers that discuss the linear regions
        %  generated by maxout activation functions. This should give some
        %  insight into what pooling does to the PWL form.
        case 'nnet.cnn.layer.AveragePooling2DLayer'      %	Average pooling layer
            report("Layer "+i+" (AveragePooling2DLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.AveragePooling3dLayer'      %	3-D average pooling layer
            report("Layer "+i+" (AveragePooling3dLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.MaxPooling2dLayer'      %	Max pooling layer
            report("Layer "+i+" (MaxPooling2dLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.MaxPooling3dLayer'      %	3-D max pooling layer
            report("Layer "+i+" (MaxPooling3dLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.MaxUnpooling2dLayer'      %	Max unpooling layer
            report("Layer "+i+" (MaxUnpooling2dLayer) is not supported yet!");
            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Addition and combination layers
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'nnet.cnn.layer.AdditionLayer'      %	Addition layer
             report("Layer "+i+" (AdditionLayer) is not supported yet!");
             
        case 'nnet.cnn.layer.ConcatenationLayer'      %	Concatenation layer
             report("Layer "+i+" (ConcatenationLayer) is not supported yet!");
             
        case 'nnet.cnn.layer.DepthConcatenationLayer'      %	Depth concatenation layer
             report("Layer "+i+" (DepthConcatenationLayer) is not supported yet!");
            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        % Output layers
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'nnet.cnn.layer.SoftmaxLayer'      %	Softmax output layer
            % Doesn't seem like anything needs to be done here, just tell
            % MATLAB how the output should be interpreted
            % report("Layer "+i+" (SoftmaxLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.ClassificationLayer'      %	Classification output layer
            % Doesn't seem like anything needs to be done here, just tell
            % MATLAB how the output should be interpreted
            % report("Layer "+i+" (ClassificationLayer) is not supported yet!");
            
        case 'nnet.cnn.layer.RegressionOutputLayer'      % Regression output layer
            % Doesn't seem like anything needs to be done here, just tell
            % MATLAB how the output should be interpreted
            % report("Layer "+i+" (RegressionOutputLayer) is not supported yet!");
            
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        % Layers that are unaccounted for, this might happen if MathWorks
        % adds new layer types.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        otherwise
            report(class(layer)+" is not accounted for by flatten_net().");    
    end
    
end

% Add the corresponding affine function to each region
for j = 1:length(regs)
    P = Ps{j};
    regs(j).addFunction(AffFunction(P(1:end-1, 1:end-1), P(1:end-1, end)), 'f');
    for i = 1:size(P, 1)-1
        regs(j).addFunction(AffFunction(P(i, 1:end-1), P(i, end)), "f"+i);
    end
    % Save local P matrix for later use
    regs(j).Data.P = P;
end

end


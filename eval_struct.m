function [y] = eval_struct(net,x)
%EVAL_STRUCT Summary of this function goes here
%   Detailed explanation goes here

y = x;
for i = 1:length(net)
    layer = net(i);
    if strcmp(layer.type, 'dense')
        y = layer.W*y + layer.b;
    else
        error('Unsupported layer type')
    end
    
    if strcmp(layer.activation, 'none')
        y = y
    if strcmp(layer.activation, 'relu')
        y = max(0,y);
    else
        error('Unsupported activation type')
    end
end


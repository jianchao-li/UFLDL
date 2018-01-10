function [out] = showBases(theta,i,varargin)
global params
if mod(i,2) == 0    
    W = reshape(theta, params.numFeatures, params.n);
    display_network(W');
end
out = 0;
end
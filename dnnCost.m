function [J grad] = dnnCost(nn_params,layers, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a L layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, layers, X, y, lambda)
%   computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters, the weight matrices
% for each layer neural network
m = size(X, 1);
L = size(layers,2);
num_labels = layers(L);
Y = sqrt(y*[1:num_labels] ) == y*ones(1,num_labels);
%%
loc = 1;
h = 0;
aVec = 0;
zVec = 0;
p = 0;
z = 0;
for i= 1: (L-1) 
    lin = layers(i);
    lout = layers(i+1);
    len = lout * (lin + 1);
    thetaVec = nn_params(loc : loc+len-1);
    theta = reshape( thetaVec, lout, lin+1);
    loc = loc + len;
    if i==1
        a = [ones(size(X,1),1) X];
        z = a*theta';
        aVec = a(:);
        zVec = z(:);
    else
        a = [ones(size(z,1),1) sigmoid(z)];
        z = a*theta';
        aVec = [ aVec; a(:)];
        if i < L-1
            zVec = [ zVec; z(:)];
        end
    end
    p = p + sum(sum(theta(:, 2:end).^2, 2));
end
aout = sigmoid(z);
h = aout;
J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m + lambda*p/(2*m);
%%
sigma = aout - Y;
loc_a = size(aVec, 1);
loc_z = size(zVec, 1);
loc_theta = size(nn_params, 1);
grad =0;
for i = L:-1:2
    lin = m;
    lout = layers(i-1)+1;
    len =  lin * lout;
    a = reshape( aVec(loc_a-len+1 : loc_a) , lin, lout);
    loc_a = loc_a - len;
    delta = (sigma'*a);
    lin = layers(i-1)+1;
    lout = layers(i);
    len =  lin * lout;
    theta = reshape( nn_params(loc_theta-len+1 : loc_theta) , lout, lin);
    loc_theta = loc_theta - len;
    p0 = (lambda/m)*[zeros(size(theta, 1), 1) theta(:, 2:end)];
    theta_grad = delta./m + p0;
    if i==L
        grad = theta_grad(:);
    else
        grad = [ theta_grad(:) ; grad(:)];
    end
    if i > 2
        len = m * layers(i-1);
        z = reshape( zVec(loc_z-len+1 : loc_z), m, layers(i-1) );
        loc_z = loc_z - len;
        sigma = (sigma*theta).*sigmoidGradient([ones(size(z, 1), 1) z]);
        sigma = sigma(:, 2:end);
    end
end

end

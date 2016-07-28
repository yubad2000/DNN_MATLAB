function [nn_params cost] = dnn(layers, X, y, lambda, itr)
fprintf('\nInitializing Neural Network Parameters ...\n')
L = size(layers,2);
initial_nn_params = 0;
for i=1:L-1
    ini_theta = randInitializeWeights(layers(i), layers(i+1));
    % Unroll parameters
    if i==1
        initial_nn_params = ini_theta(:);
    else
        initial_nn_params = [initial_nn_params(:) ; ini_theta(:)];
    end
end
%%
fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', itr);
% Create var name for the cost function to be minimized
costFunction = @(p) dnnCost(p, layers, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
end
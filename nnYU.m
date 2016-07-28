load('DigitData.mat');
load('weights.mat')
m = size(X, 1);
layers = [400 100 25 10];
lambda = 0.5;
[nn_params cost] = dnn(layers, X, y, lambda, 100);
% get the first 2 layer's parameters Theta1 and Theta2 back from nn_params
loc=1;
len = layers(2) * (layers(1) + 1);
Theta1 = reshape(nn_params(loc:loc+len-1), ...
                 layers(2), (layers(1) + 1));
loc=loc+len;
len = layers(3) * (layers(2) + 1);
Theta2 = reshape(nn_params(loc:loc+len-1), ...
                 layers(3), (layers(2) + 1));
%%
fprintf('\nVisualizing Neural Network... \n')
displayData(Theta1(:, 2:end));
%%
pred = predictYU(nn_params, layers, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
pause;
for i = 1:m
    % Display
    if (pred(i) ~=  y(i)) 
        displayData(X(i, :));
        fprintf('\nNeural Network Prediction: %d (digit %d)\n', mod(pred(i),10), mod(y(i),10) );
        pause;
    end 
end

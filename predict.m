function p = predict(nn_params, layers, X)
m = size(X, 1);
L = size( layers, 2);
loc=1;
for i=1:L-1
    len = layers(i+1) * (layers(i) + 1);
    theta = reshape(nn_params(loc:loc+len-1), ...
                 layers(i+1), (layers(i) + 1));
    loc = loc + len; 
    if i==1
        h = sigmoid([ones(m, 1) X] * theta');
    else
        h = sigmoid([ones(m, 1) h] * theta');
    end
end
[dummy, p] = max(h, [], 2);

end

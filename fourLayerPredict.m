function p = fourLayerPredict(unrolledTheta, hidden_layer1_size, hidden_layer2_size, input_layer_size, num_labels, X)

Theta1 = reshape(unrolledTheta(1 : hidden_layer1_size * (input_layer_size + 1)),...
                 hidden_layer1_size, (input_layer_size + 1));


Theta2 = reshape(unrolledTheta(hidden_layer1_size * (input_layer_size + 1) + 1 : (hidden_layer1_size * (input_layer_size + 1)) + (hidden_layer2_size * (hidden_layer1_size + 1))),...
                  hidden_layer2_size, (hidden_layer1_size + 1));

Theta3 = reshape(unrolledTheta((hidden_layer1_size * (input_layer_size + 1)) + (hidden_layer2_size * (hidden_layer1_size + 1) + 1) : end),...
                  num_labels, (hidden_layer2_size + 1));

m = size(X, 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
h3 = sigmoid([ones(m, 1) h2] * Theta3');
size(h3);
[v, p] = max(h3, [], 2);
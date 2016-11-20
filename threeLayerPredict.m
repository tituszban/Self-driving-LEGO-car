function p = threeLayerPredict(unrolledTheta, input_layer_size, hidden_layer_size, num_labels, X)

Theta1 = reshape(unrolledTheta(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(unrolledTheta(hidden_layer_size * (input_layer_size + 1) + 1:end),num_labels, (hidden_layer_size + 1));

m = size(X, 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[V, p] = max(h2, [], 2);
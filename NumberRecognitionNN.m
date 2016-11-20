clear; clc

load ex4data1.mat;

input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;

initial_Theta1 = randInitialThetas(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitialThetas(hidden_layer_size, num_labels);

unrolledThetas = [initial_Theta1(:); initial_Theta2(:)];

numericalGradientCheck(3);

options = optimset('MaxIter', 50);
lambda = 0.01;

costFunction = @(p) CostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

[Thetas, cost] = fmincg(costFunction, unrolledThetas, options);

Predicted = predict(Thetas, hidden_layer_size, input_layer_size, num_labels, X);

mean(double(Predicted == y)) * 100
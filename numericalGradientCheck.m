function numericalGradientCheck(lambda)

input_layer_size = 4;
hidden_layer_size = 5;
num_labels = 3;

m = 6;

Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
X = debugInitializeWeights(m, input_layer_size - 1);
y = 1 + mod(1:m, num_labels)';

unrolledTheta = [Theta1(:); Theta2(:)];

costFunc = @(p) CostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

[cost grad] = costFunc(unrolledTheta);

numgrad = zeros(size(unrolledTheta));
perturb = zeros(size(unrolledTheta));
e = 1e-4;

for i = 1:numel(unrolledTheta)
  perturb(i) = e;
  loss1 = costFunc(unrolledTheta - perturb);
  loss2 = costFunc(unrolledTheta + perturb);
  numgrad(i) = (loss2 - loss1) / (2*e);
  perturb(i) = 0;
end

disp([numgrad grad]);

diff = norm(numgrad - grad)/norm(numgrad + grad)

end
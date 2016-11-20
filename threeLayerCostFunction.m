function [J grad] = threeLayerCostFunction(unrolledTheta, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)



Theta1 = reshape(unrolledTheta(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(unrolledTheta(hidden_layer_size * (input_layer_size + 1) + 1:end), num_labels, (hidden_layer_size + 1));

m = size(X, 1);

s = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for i = 1:m
  a1 = [1; X(i,:)'];
  z2 = Theta1 * a1;
  a2 = [1; sigmoid(z2)];
  z3 = Theta2 * a2;
  h = sigmoid(z3);
  
  Y = zeros(num_labels, 1);
  Y(y(i)) = 1;
  
  try
    s = s + sum(-Y' * log(h) - ((1-Y') * log(1 - h)));
  catch
    size(Y)
    size(h)
  end_try_catch
  
  %backprop
  d3 = h - Y;
  d2 = (Theta2' * d3)(2:end) .* sigmoidGradient(z2);
  
  Theta1_grad = Theta1_grad + d2 * a1';
  Theta2_grad = Theta2_grad + d3 * a2';
endfor;

t = sum(sum(Theta1(1:hidden_layer_size, 2:input_layer_size+1) .^ 2, 2)) + sum(sum(Theta2(1:num_labels, 2:hidden_layer_size + 1) .^2, 2));

J = (1/m) * s + (lambda / (2 * m)) * t;


Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

%Theta1_grad(1:size(Theta1_grad,1), 2:size(Theta1_grad,2)) = (1 + lambda / m) * Theta1_grad(1:size(Theta1_grad,1), 2:size(Theta1_grad,2));
%Theta2_grad(1:size(Theta2_grad,1), 2:size(Theta2_grad,2)) = (1 + lambda / m) * Theta2_grad(1:size(Theta2_grad,1), 2:size(Theta2_grad,2));

for i = 1:size(Theta1_grad, 1)
  for j = 2:size(Theta1_grad, 2)
    Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda / m) * Theta1(i,j);
  end
end

for i = 1:size(Theta2_grad, 1)
  for j = 2:size(Theta2_grad, 2)
    Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda / m) * Theta2(i,j);
  end
end


grad = [Theta1_grad(:); Theta2_grad(:)]; 

end
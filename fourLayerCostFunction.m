function [J grad] = fourLayerCostFunction(unrolledTheta, input_layer_size, hidden_layer1_size, hidden_layer2_size, num_labels, X, y, lambda)



Theta1 = reshape(unrolledTheta(1 : hidden_layer1_size * (input_layer_size + 1)),...
                 hidden_layer1_size, (input_layer_size + 1));


Theta2 = reshape(unrolledTheta(hidden_layer1_size * (input_layer_size + 1) + 1 : (hidden_layer1_size * (input_layer_size + 1)) + (hidden_layer2_size * (hidden_layer1_size + 1))),...
                  hidden_layer2_size, (hidden_layer1_size + 1));

Theta3 = reshape(unrolledTheta((hidden_layer1_size * (input_layer_size + 1)) + (hidden_layer2_size * (hidden_layer1_size + 1) + 1) : end),...
                  num_labels, (hidden_layer2_size + 1));

m = size(X, 1);

s = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

for i = 1:m
  a1 = [1; X(i,:)'];
  z2 = Theta1 * a1;
  a2 = [1; sigmoid(z2)];
  z3 = Theta2 * a2;
  a3 = [1; sigmoid(z3)];
  z4 = Theta3 * a3;
  h = sigmoid(z4);
  
  
  Y = zeros(num_labels, 1);
  Y(y(i)) = 1;
  
  s = s + (-Y' * log(h) - ((1-Y') * log(1 - h)));
  
  %backprop
  d4 = h - Y;
  d3 = (Theta3' * d4)(2:end) .* sigmoidGradient(z3);
  d2 = (Theta2' * d3)(2:end) .* sigmoidGradient(z2);
  
  Theta1_grad = Theta1_grad + d2 * a1';
  Theta2_grad = Theta2_grad + d3 * a2';
  Theta3_grad = Theta3_grad + d4 * a3';
end;

s1 = sum(sum(Theta1(1:hidden_layer1_size, 2:input_layer_size+1) .^ 2, 2));
s2 = sum(sum(Theta2(1:hidden_layer2_size, 2:hidden_layer1_size + 1) .^2, 2));
s3 = sum(sum(Theta3(1:num_labels, 2:hidden_layer2_size + 1) .^2, 2));

t = s1 + s2 + s3;

J = (1/m) * s + (lambda / (2 * m)) * t;


Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;
Theta3_grad = (1/m) * Theta3_grad;

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

for i = 1:size(Theta3_grad, 1)
  for j = 2:size(Theta3_grad, 2)
    Theta3_grad(i,j) = Theta3_grad(i,j) + (lambda / m) * Theta3(i,j);
  end
end

grad = [Theta1_grad(:); Theta2_grad(:); Theta3_grad(:)]; 

end
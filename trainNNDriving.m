%clear;

log = csvread('processed_log.txt');

m = size(log, 1);

X = log(:, 1:600); % log(:, 1:600);% [ones(m, 1) log(:, 1:1600)];
Y_steer = round((log(:, 601) + 1) * 15 + 1);
%Y_speed = round((log(:, 1602) + 1) * 50);

input_layer_size = 600;
hidden_layer1_size = 133;
%hidden_layer2_size = 30;
num_labels = 31;

initial_Theta1 = randInitialThetas(input_layer_size, hidden_layer1_size);
%initial_Theta2 = randInitialThetas(hidden_layer1_size, hidden_layer2_size);
%initial_Theta3 = randInitialThetas(hidden_layer2_size, num_labels);
initial_Theta2 = randInitialThetas(hidden_layer1_size, num_labels);

%unrolledThetas = [initial_Theta1(:); initial_Theta2(:); initial_Theta3(:)];
%unrolledThetas = [initial_Theta1(:); initial_Theta2(:)];

load Thetas.txt
unrolledThetas = Thetas;
clear('Thetas')

%numericalGradientCheck(3);

options = optimset('MaxIter', 200);
lambda = 0.01;

%printf "Start optim\n"

%cFunction = @(p) fourLayerCostFunction(p, input_layer_size, hidden_layer1_size, hidden_layer2_size, num_labels, X, Y_steer, lambda);
cFunction = @(p) threeLayerCostFunction(p, input_layer_size, hidden_layer1_size, num_labels, X, Y_steer, lambda);

[Thetas, cost] = fmincg(cFunction, unrolledThetas, options);

save Thetas.txt Thetas

%Predicted1 = fourLayerPredict(Thetas, hidden_layer1_size, hidden_layer2_size, input_layer_size, num_labels, X);
Predicted1 = threeLayerPredict(Thetas, input_layer_size, hidden_layer1_size, num_labels, X);

printf('training predicted: ')
mean(double(Predicted1 == Y_steer)) * 100

%use TestData
#{
testX = testData(:, 1:600);
testY = round((testData(:, 601) + 1) * 15 + 1);

PredictedTest = threeLayerPredict(Thetas, input_layer_size, hidden_layer1_size, num_labels, testX);
printf('test predicted: ')
mean(double(PredictedTest == testY)) * 100
#}
load log.txt;

m = size(log, 1);

X = [ones(m, 1) log(:, 1:1600)];
Y_steer = log(:, 1601);
Y_speed = log(:, 1602);

Theta_steer = pinv(X' * X) * X' * Y_steer;
save Tehta_steer.txt Theta_steer;

Theta_speed = pinv(X' * X) * X' * Y_speed;
save Theta_speed.txt Theta_speed;
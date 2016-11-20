function [steer, speed] = evaluateLinearDriving(theta_speed, theta_steer, x)

x = [1, x];

steer = x * theta_steer;
speed = x * theta_speed;
clc;
close all;
clear;

%%%%%%%% PART1: Using Multi-class Logistic Regression

input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;  % 10 labels, from 1 to 10, we have mapped "0" to label "10")

load('TrainingData.mat');  % Loading training data (they will be stored in arrays X, y)
m = size(X, 1);  % Number of training examples

fprintf('\nTesting: '); % First we test the program
% Assigning the parameters to some values
theta_t = [-2; -1; 1; 2];
X_t = [ones(5, 1) reshape(1:15, 5, 3)/10];
y_t = ([1; 0; 1; 0; 1] >= 0.5);
lambda_t = 3;

% Computing the cost and gradients using rCostFunction
[J, grad] = rCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

% Then, we train a one-vs-all multi-class Logistic Regression classifier
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;  % Defining the value of lambda
% Finding the all theta parameters for each class (the last parameter is
% the number of iterations)
[all_theta] = genThetas(X, y, num_labels, lambda, 50);  

% Now we use our trained classifer to predict for all examples in data set
% (In this part, the input matrix can be any new matrix containing images
% of digits with size of 20x20 pixels)
inputMatrix = X;
m = size(inputMatrix, 1);
num_labels = size(all_theta, 1);
p = zeros(size(inputMatrix, 1), 1);
Xn = [ones(m, 1) X];  % Adding ones to the X data matrix
pred = sigmoid(Xn*all_theta');
[maxP, indP] = max(pred, [], 2);
pred = indP;

% Finally, we find the accuracy of our classifier
acc = mean(double(pred == y)) * 100;
fprintf('\nTraining Set Accuracy: %f\n', acc);

%% ---------------------------------------------------------------------------------

%%%%%%%% PART2: Using Feedforward Propagation in a Pre-trained Neural Network

clc;
close all;
clear;

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 Hidden units
num_labels = 10;  % 10 labels, from 1 to 10, we have mapped "0" to label "10")

load('TrainingData.mat');  % Loading training data (they will be stored in arrays X, y)
m = size(X, 1);  % Number of training examples

% Loading some pre-initialized neural network parameters.
load('NNweights.mat');  % Weights will be loaded into variables Theta1 and Theta2

% Now that we have trained the neural network, we would like to use it 
% to predict the digits (The input matrix can be any new matrix containing 
% images of digits with size of 20x20 pixels)
inputMatrix = X;
m = size(inputMatrix, 1);
num_labels = size(Theta2, 1);
p = zeros(size(inputMatrix, 1), 1);
Xn = [ones(m, 1) X];  % Adding ones to the X data matrix
newX = sigmoid(Xn*Theta1');
newX = [ones(m, 1) newX];
pred2 = sigmoid(newX*Theta2');
[maxP, indP] = max(pred2, [], 2);
pred2 = indP;

% Finally, we find the accuracy of our classifier
acc2 = mean(double(pred2 == y)) * 100;
fprintf('\nTraining Set Accuracy: %f\n', acc2);

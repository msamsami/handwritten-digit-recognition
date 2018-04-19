clc;
close all;
clear;

%%%%%%%% PART1: Using Multi-class Logistic Regression

input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;  % 10 labels, from 1 to 10 (we have mapped "0" to label "10")

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

% Finding the all theta parameters for each class (the last parameter of
% genThetas, i.e., the one which equals to 50, defines the maximum number
% of iterations
[all_theta] = genThetas(X, y, num_labels, lambda, 50);  

% Now we use our trained classifer to predict for all examples in data set
% (In this part, the input matrix can be any new matrix containing images
% of digits with size of 20x20 pixels)
inputMatrix = X;
m = size(inputMatrix, 1);
num_labels = size(all_theta, 1);
p = zeros(size(inputMatrix, 1), 1);
Xn = [ones(m, 1) inputMatrix];  % Adding ones to the X data matrix
pred = sigmoid(Xn*all_theta');
[maxP, indP] = max(pred, [], 2);
pred = indP;

% Finally, we find the accuracy of our classifier
acc = mean(double(pred == y)) * 100;
fprintf('\nTraining Set Accuracy: %f\n', acc);

%% ---------------------------------------------------------------------------------

%%%%%%%% PART2: Using Feedforward Propagation for a Pre-trained Neural Network

%clc;
%close all;
clear;

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 Hidden units
num_labels = 10;  % 10 labels, from 1 to 10 (we have mapped "0" to label "10")

load('TrainingData.mat');  % Loading training data (they will be stored in arrays X, y)
m = size(X, 1);  % Number of training examples

% Loading some pre-initialized neural network parameters.
load('NNweights.mat');  % Weights will be loaded into variables Theta1 and Theta2

% After training the neural network, we would like to use it to predict
% the digits (The input matrix can be any new matrix containing images
% of digits with size of 20x20 pixels)
inputMatrix = X;
pred2 = NNpredict(Theta1, Theta2, inputMatrix);

% Finally, we find the accuracy of our classifier
acc2 = mean(double(pred2 == y)) * 100;
fprintf('\nTraining Set Accuracy (Pre-trained Neural Network Method): %f\n', acc2);

%% ---------------------------------------------------------------------------------

%%%%%%%% PART3: Using Backpropagation for Neural Network Learning

%clc;
%close all;
clear;

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 Hidden units
num_labels = 10;  % 10 labels, from 1 to 10 (we have mapped "0" to label "10")

load('TrainingData.mat');  % Loading training data (they will be stored in arrays X, y)
m = size(X, 1);  % Number of training examples

% Loading some pre-initialized neural network parameters.
load('NNweights.mat');  % Weights will be loaded into variables Theta1 and Theta2

nn_params = [Theta1(:) ; Theta2(:)];  % Unrolling parameters


lambda = 1;  % Weight regularization parameter (we set this to 1 here)

% Computing the cost
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda); 

% Initializing pameters
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];  % Unrolling initialized parameters

checkNNGradients;  % Checking gradients by running checkNNGradients function

% Once we made sure that backpropagation implementation is correct,
% we should continue to implement the regularization with the cost and gradient.
lambda = 3;
checkNNGradients(lambda);  % Checking gradients by running checkNNGradients function

% Also we output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);  
           
           
% We have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". These advanced optimizers
%  are able to train our cost functions efficiently as long as we provide
%  them with the gradient computations.

options = optimset('MaxIter', 75);  % Maximum number of iterations

lambda = 1;  % We should also try different values of lambda

% Creating "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
% Now that we have trained the neural network, we would like to use it 
% to predict the digits (The input matrix can be any new matrix containing 
% images of digits with size of 20x20 pixels)
inputMatrix = X;
pred3 = NNpredict(Theta1, Theta2, inputMatrix);

% Finally, we find the accuracy of our classifier
acc3 = mean(double(pred3 == y)) * 100;
fprintf('\nTraining Set Accuracy (Backpropagation Method): %f\n', acc3);

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

%neural network which performs classification
%   This function computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.


% Reshaping nn_params back into the parameters Theta1 and Theta2, the 
% weight matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X];

for i = 1:m
    z2 = X(i, :)*Theta1';
    a2 = [1 sigmoid(z2)];  % Activation of the hidden layer
    z3 = (a2)*Theta2';
    h = sigmoid(z3);  % Activation of the output layer, i.e., the hypothesis
    yi = zeros(num_labels, 1);
    yi(y(i), 1) = 1;  % True target value
    delta3 = h' - yi;
    delta2 = (Theta2'*delta3).*([1; sigmoidGradient(z2)']);
    delta2 = delta2(2:end);
    Theta2_grad = Theta2_grad + delta3*a2; 
    Theta1_grad = Theta1_grad + delta2*X(i, :);
    J = J + sum(-yi.*log(h') - (1-yi).*log(1-h'));
end

J = J/m;  % Cost, without regularization
J = J + (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end).^2)) + ...
    sum(sum(Theta2(:, 2:end).^2)));  % Cost, including regularization term

Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

Theta2_grad = (Theta2_grad/m) + (lambda/m)*Theta2;
Theta1_grad = (Theta1_grad/m) + (lambda/m)*Theta1;

% Unrolling gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

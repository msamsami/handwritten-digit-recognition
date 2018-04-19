function g = sigmoidGradient(z)
    % This function returns the gradient of the sigmoid function
    g = zeros(size(z));
    g = sigmoid(z).*(1 - sigmoid(z));
end

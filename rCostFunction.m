function [J, grad] = rCostFunction(theta, X, y, lambda)

    m = length(y);  % Number of training examples
    grad = zeros(size(theta));

    ntheta = theta(2:end, :);  % Excluding the first value, i.e., theta0
    nX = X(:, 2:end);  % Excluding the first column

    J = sum(-y.*log(sigmoid(X*theta)) - (1-y).*log(1-sigmoid(X*theta)))/m ...
        + (lambda/(2*m))*(ntheta'*ntheta);

    grad = (1/m)*(nX'*(sigmoid(X*theta)-y)) + (lambda/m)*ntheta;
    grad = [(((X(:, 1))'*(sigmoid(X*theta)-y))/m); grad];

end

function p = NNpredict(Theta1, Theta2, X)

    % Useful values
    m = size(X, 1);
    num_labels = size(Theta2, 1);
    p = zeros(size(X, 1), 1);

    % Add ones to the X data matrix
    X = [ones(m, 1) X];

    newX = sigmoid(X*Theta1');
    newX = [ones(m, 1) newX];
    p = sigmoid(newX*Theta2');
    [maxP, indP] = max(p, [], 2);
    p = indP;

end

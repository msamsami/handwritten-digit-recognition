function [all_theta] = genThetas(X, y, num_labels, lambda, numIterations)

    % Some useful variables
    m = size(X, 1);  % Number of training examples
    n = size(X, 2);  % Number of parameters for each class (equals to LengthOfImages*WidthOfImages)

    all_theta = zeros(num_labels, n + 1);
    X = [ones(m, 1) X];  % Adding ones to the X data matrix

    for i = 1:num_labels
        initial_theta = zeros(n + 1, 1);
        options = optimset('GradObj', 'on', 'MaxIter', numIterations);
        temp = fmincg(@(t)(rCostFunction(t, X, (y == i), lambda)), ...
                  initial_theta, options);
        all_theta(i, :) = temp';
    end

end

function W = randInitializeWeights(L_in, L_out)
%   This function randomly initializes the weights of a layer with L_in
%   incoming connections and L_out outgoing connections.
epsiloninit = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsiloninit - epsiloninit;
end

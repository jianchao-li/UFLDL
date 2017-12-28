function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
prev = data;
for i = 1 : numel(hAct)
    z = stack{i}.W * prev + stack{i}.b;
    if (i == numel(hAct))
        hAct{i} = bsxfun(@rdivide, exp(z), sum(exp(z)));
    else
        switch ei.activation_fun
            case 'logistic'
                hAct{i} = 1 ./ (1 + exp(-z));
            case 'tanh'
                hAct{i} = (exp(z) - exp(-z)) ./ (exp(z) + exp(-z));
            case 'relu'
                hAct{i} = max(z, 0);
        end
    end
    prev = hAct{i};
end
pred_prob = hAct{end};

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end

%% compute cost
%%% YOUR CODE HERE %%%
m = size(data, 2);
logs = log(hAct{end}');
inds = sub2ind(size(logs), 1 : size(logs, 1), labels');
ceCost = -1 / m * sum(logs(inds));

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
delta = cell(numHidden + 1, 1);
delta{end} = -(full(sparse(labels, 1 : size(labels, 1), 1)) - hAct{end});
for i = numel(delta) - 1 : -1 : 1
    delta{i} = stack{i + 1}.W' * delta{i + 1} .* (hAct{i} .* (1 - hAct{i}));
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for i = 1 : numel(stack)
    wCost = wCost + sum(stack{i}.W(:) .^ 2);
end
wCost = ei.lambda / 2 * wCost;
cost = ceCost + wCost;
prev = data;
for i = 1 : numel(gradStack)
    gradStack{i}.W = 1 / m * (delta{i} * prev') + ei.lambda * stack{i}.W;
    prev = hAct{i};
end
for i = 1 : numel(gradStack)
    gradStack{i}.b = 1 / m * sum(delta{i}, 2);
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end




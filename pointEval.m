%% Calculates the network and the Jacobian of output w.r.t. inputs
% written by ~, last modified Nov. 24, 2018.
%
% NN stands for Neural Network
%
% Inputs:
    % 1- <x> is a single input to the NN
    % 2- <net> is a struct that contains the trained network
    % 3- <calcJac> is a binary parameter defining wheter the Jacobian w.r.t input 
    %    should be computed or not
    % 4- <print> is a binary parameter, which makes the function print its output
    % 5- <soft> is a binary parameter, defining whether to apply a softmax to the
    %    output of NN
%
% Outputs:
% output is a struct named pdata (point data because it is specific to a single x)
    % <pdata> has saveral fields:
        % pdata.x is the input x to NN
        % pdata.y is a cell array containing the output of each inner layer
        % pdata.z is the output of the NN without softmax
        % pdata.zs is the softmax output of NN - computed if soft=1
        % pdata.J is a cell array containing the Jacobian of output of each 
        %     layer w.r.t input. pdata.J{end} is the Jacobian of output.
        % pdata.sigmaNN is the sigmaNN copied to pdata too

function [pdata] = pointEval(x,net,calcJac,print_output,soft)
    if nargin < 3;  calcJac = 1; end
    if nargin < 4;  print_output = 0; end
    if nargin < 5;  soft = 1; end

    pdata = calc_output(x,net);
    if calcJac % parameter whether to return the Jacobian or not
        pdata.J = calc_Jac(x,net);
    %     pdata.Jnum = JacNum(x,sigmaNN,net,0.0001); for checking the J
    end
    if soft
        pdata.zs = pointSoftEval(pdata.z);
    end
    if print_output
        display(pdata.z, 'output of network');
        if soft
            display(pdata.zs, 'softmax output of network');
        end
    end
end

%% calculating the output
function [pdata] = calc_output(x,net)
    % copy x
    pdata.x = x; 
    % copy net.sigmaNN
    pdata.sigmaNN = net.sigmaNN; 
    % compute output of 1st layer
    pdata.y{1} = erf((x * net.w{1} + net.b{1}) ./ net.sigmaNN(1));
    % compute output of middle layers
    for i=2:net.nl
        pdata.y{i} = erf((pdata.y{i-1} * net.w{i} + net.b{i}) ./ net.sigmaNN(i));
    end
    % compute output of network
    pdata.z = pdata.y{net.nl} * net.w{net.nl+1} + net.b{net.nl+1};
end

%% deriving Jacobians on each level (including the output layer) 
% with respect to the input
function [J_y] = calc_Jac(x,net)
    % setting empty containers
    J_y{net.nl} = [];  e{net.nl} = [];
    % obtain the calculated network
    pdata = calc_output(x,net);
    % compute Jac for the 1st layer
    e{1} = exp(-((x * net.w{1} + net.b{1})./(net.sigmaNN(1)))'.^2);
    J_y{1} = (2/sqrt(pi)/net.sigmaNN(1)) .* (e{1} * ones(1,net.nx)) .* net.w{1}';
    % compute Jac for the layers 2 through net.nl
    for i = 2:net.nl
        e{i} = exp(-((pdata.y{i-1} * net.w{i} + ... 
            net.b{i})./(sqrt(1) * net.sigmaNN(i)))'.^2); % intermediate parameter
        J_y{i} = (2/sqrt(pi)/net.sigmaNN(i)) * (e{i} * ones(1,net.nx)) .* ...
            (net.w{i}' * J_y{i-1});
    end
    % compute Jac for the output layer
    J_y{net.nl+1} = net.w{net.nl+1}' * J_y{net.nl};
end

%% softmax function
function [softout] = pointSoftEval(out)
    softout = exp(out)./(sum(exp(out)));
end

%%%%%%%%%%%%%%%%%%%%%%
%% Archived functions

% numerical Jacobian of output with respect to input using forward difference
function [numJ_y] = JacNum(x,net,h)
    if nargin < 4
        h = 0.001;
    end
    numJ_y{net.nl} = [];
    data_0 = calc_output(x,net.sigmaNN,net);
    for j = 1:net.nx % for each input
        x_frwd = x + h .* sparse(1,j,1,1,net.nx);
        data_1 = calc_output(x_frwd,net.sigmaNN,net);
        for i = 1:net.nl
            numJ_y{i}(:,j) = (data_1.y{i} - data_0.y{i})'./h;
        end
        % this is the numerical Jacobian for the output layer
        numJ_y{net.nl+1}(:,j) = (data_1.z - data_0.z)'./h;
    end
end
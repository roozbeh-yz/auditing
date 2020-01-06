%% Computes the closest flip point to an input
% written by ~, last modified Dec. 12, 2018.
%
% Inputs
    % 1- <x_hat> is a single input to the network
    % 2- <this> is the class associated with x_hat
    % 3- <that> is the class that we want to flip to
    % 4- <net> is a struct that contains the trained NN
    % 5- <x0> the starting point - if not provided it will be set equal to x
    % 6- <mybounds> is a struct containing two scalars or vectors with same size as x,
        % mybounds.low defines the lower bound, and 
        % mybounds.up defines the upper bound.
        % if not provided, the lower bound will be automatically set to
        % zero and upper bound will be set to 1e4.
%
% Outputs
    % 1- <xf> is the closest flip point
    % 2- <c> is the softmax output that has been equalized for this and that classes
        % c will be NaN if the soft output of classes is unequal
    % 3- <dist> is the Euclidean distance between the x_hat and its closest flip point
    % 4- <elap_time> is the total time spent to find the xf

function [xf,c,dist,elap_time] = CFP(x_hat,this,that,net,x0,mybounds)
    tic
    if nargin < 5; x0 = 1.0 .* x_hat; end

    if nargin < 6 || isempty(mybounds)
        opt.lower_bounds = 0 .* ones(size(x0)); % same as below
        opt.upper_bounds = 1e4 .* ones(size(x0)); % depends on the dataset
    else
        opt.lower_bounds = mybounds.low .* ones(size(x0));
        opt.upper_bounds = mybounds.up .* ones(size(x0));
    end
    if sum(x0 > opt.upper_bounds) > 0 || sum(x0 < opt.lower_bounds) > 0
        x1 = x0;
        x0 = min(x0,opt.upper_bounds); x0 = max(x0,opt.lower_bounds);
        fprintf('----Starting point outside the bounds! norm = %.4f----\n',norm(x0-x1,2));
    end

    if net.nz > 2
        otherList = setdiff(1:net.nz,[this,that]);
    elseif net.nz < 2
        fprintf('At least two output classes needed to find a flip between them.\n');
    end

    % objective function
    fun = @(x)fung(x,x_hat);

    % nonlinear constraint
    nonlcon = @(x)mycon(x,this,that,net);
    % nonlcon = @(x) eqCon(x,this,that,net);
    % nonlcon = [];

    % bounds
    lb = opt.lower_bounds; % lower bound
    ub = opt.upper_bounds; % upper bound
    options = optimoptions('fmincon','Display','none','Algorithm','interior-point',...
        'SpecifyObjectiveGradient',true, ...
        'SpecifyConstraintGradient',true,'MaxIterations',500);

    % optimize using fmincon function in MATLAB
    [xf,~,retcode,~] = fmincon(fun,x0,[],[],[],[],lb,ub,nonlcon,options);

    % evaluate the xf and check if it is a flip
    pdata = pointEval(xf,net,0);
    if (abs(pdata.z(this) - pdata.z(that)) < 1e-4)
        c = min(pdata.zs(this),pdata.zs(that));
    else
        c = nan;
        fprintf('>unsuccessful: %.4f uneq %.4f \n', pdata.zs(this) , pdata.zs(that));
    end

    % check all other classes have smaller score - for multiclass models
    if net.nz > 2
        if sum(pdata.zs(this)<=pdata.zs(otherList)) > 0
            c = nan;
            fprintf('>inequality constrains not satisfied %.4f < %.4f \n', ...
                pdata.zs(this), max(pdata.zs(otherList)));
        end
    end
    % check the retcode
    if retcode < 0
        c = nan;
        fprintf('>no solution found as retcode is %d < 0.\n', retcode);
    end
    dist = norm(xf-x_hat,2);
    pointEval(xf,net,0,0,1);
    % fprintf('c: %.4f and dist: %.4f \n',c,dist);
    elap_time = toc;

end
%% calculates least squares and its derivative -- objective function for CFP
function [f,g] = fung(x,x_hat)
    f = norm(x-x_hat,2)^2;
    g = 2.*(x-x_hat);
end

%% equality constraint
function [ceq,gceq] = eqCon(x,this,that,net)
    pdata = pointEval(x,net,1);
    ceq = pdata.z(this) - pdata.z(that);  % Compute nonlinear equalities at x
    gceq = pdata.J{end}(this,:)' - pdata.J{end}(that,:)';
end

%% inequality constraints
function [c,gc] = ineqCon(x,net,this,otherList)
    pdata = pointEval(x,net,1);
    c = pdata.z(otherList) - pdata.z(this) + 0.2/net.nz;
    gc = pdata.J{end}(otherList,:)' - pdata.J{end}(this,:)'*ones(1,length(otherList));
end

%% main nonlinear constraint function, including 
% the equality and inequality constraints
function [c,ceq,gc,gceq] = mycon(x,this,that,net)
    [ceq,gceq] = eqCon(x,this,that,net);
    otherList = setdiff(1:net.nz,[this,that]);
    [c,gc] = ineqCon(x,net,this,otherList);
end

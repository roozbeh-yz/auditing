%% Function to calculate the closest flip points for a set of data points 
% written by ~, last modified August 22, 2019.
%
% NN: neural network , CFP: closest flip point
%
% Inputs:
    % 1- <strList> a struct produced by appNNtoData containing information about data points:
        % strList.raw contains the essential information. Its rows correspond
        % to some or all of rows in myD:
            % 1st column is the row index of data point in myD
            % 2nd column is the correct label for data point
            % 3rd column is the other label for calculating the flip point
            % 4th column is the softmax value for the prediction of model
    % 2- <net> is the trained NN
    % 3- <myD> is a struct containing a set of datapoints (myD.X) and labels (myD.Y)
    % 4- <processlist> is the list of rows in strList that we want to be processed by the
        % function. If processlist is empty, the function will process all the rows
        % in strList.
    % 5- <startp> is a parameter defining the starting point as x0 = x.*startp,
        % where x is the individual data points
        % startp is either a scalar or a vector with same size as any data
        % point.
        % If size of processlist is 1 and startp is a vector, startp will 
        % be treated as the actual starting point x0.
    % 6- <mybounds> is a struct for the upper and lower bounds passed to CFP.
    % 7- <homo> binary parameter defining whether the CFP_homo should be used
        % instead of CFP.
%
% Outputs:
    % <strList> same struct with additional information about the closest flip points
      % The additional information includes:
        % columns 5 through 7 in strList.raw:
            % 5th column is the time spent to compute the closest flip point
            % 6th column is the value of c (defined in CFP)
            % 7th column is the distance between the points and their closest
                % flip points
        % strList.flips containd the flip points corresponding to each row in
            % strList.raw, its columns represent the features, same as myD.


function [strList] = flips_calc(strList,net,myD,processlist,startp,mybounds,homo)
    % creating the field for flip points
    nmistakes = size(strList.raw,1);
    nfeatures = size(myD.X,2);

    % set up the field for flip points, strList.flips
    if ~isfield(strList,'flips')
        strList.flips = zeros(nmistakes,nfeatures);
    elseif sum(size(strList.flips) == [nmistakes,nfeatures]) < 2
        strList.flips = zeros(nmistakes,nfeatures);
    end

    % setting the list of mistakes to process
    if isempty(processlist)
        calc_list = 1:size(strList.raw,1);
    else
        calc_list = processlist;
    end
    if length(processlist) > 1
        fprintf('---- starting flips_calc function for several ---- \n');
    end

    % other parametrs in case they are passed in empty
    if nargin < 7; homo = []; end
    if nargin < 6; mybounds = []; end
    if nargin < 5 || isempty(startp); startp = 1.0; end

    % make sure there are 7 columns in strList.raw
    if size(strList.raw,2) < 7
        strList.raw(end,7) = 0;
    end

    prevtime = 0; % variable to keep track of computational time, when an 
                  % existing flip point is improved, the new computation time
                  % is added to the previous computation time.

    myprint = 1;  % if someone does not want the progress to be printed, it 
                  % can be changed to 0.

    % looping over the list of points and finding their closest flip points
    k = 0; % counter
    for i = calc_list
        idx = strList.raw(i,1);
        k = k + 1;
        if myprint; fprintf('--- processing element %d , Lrow %d - %d - \n',idx,i,k); end
        x = myD.X(idx,:);

        % defining the 2 classes that we want to flip between
        label_this = strList.raw(i,2);
        label_that = strList.raw(i,3);
        % search for the closest flip point
        if size(startp,2) == size(myD.X,2) && length(processlist) == 1
            x0 = startp; prevtime = strList.raw(i,5);
        else
            x0 = x .* startp;
        end
        tic
        if isempty(homo)
            [xf,c,dist,sptime] ...
                = CFP(x,label_this,label_that,net,x0,mybounds);
        else
            sigma_homo = homo.sigma_homo;
            [xf,c,dist,sptime] ...
                = CFP_homo(x,label_this,label_that,net,x0,mybounds,homo.niter,sigma_homo);
        end
        savepoint = 1; % this parameter can be set to zero, if one wants to 
                       % only see the progress and does not want to record 
                       % the flip point information
                       
        % check if the newly found point is better than the one found before
        if strList.raw(i,7) > 0 && strList.raw(i,6) > 0
            if isnan(c)
                fprintf(' - Could not find a flip point. Did not save it.\n');  savepoint = 0;
            elseif dist >= strList.raw(i,7)
                fprintf(' - New point not saved - %.4f not closer than %.4f saved before.\n',...
                    dist,strList.raw(i,7));   savepoint = 0;
            end
        end
        if savepoint
            if ~isnan(c)
                if myprint; fprintf(' -- saving new point -improved dist = %.5f to %.5f - c = %.3f to %.3f --\n'...
                    ,strList.raw(i,7),dist,strList.raw(i,6),c); end
            else
                fprintf(' -- failed but saving new point - dist = %.5f to %.5f - c = %.3f to %.3f --\n'...
                    ,strList.raw(i,7),dist,strList.raw(i,6),c);
            end
            strList.raw(i,5) = sptime + prevtime;
            strList.raw(i,6) = c;
            strList.raw(i,7) = dist;
            strList.flips(i,:) = xf;
        end
        % print the progress every 100 iterations
        if ~myprint && mod(i,100)==0; fprintf('calculated %d flips.\n',i); end
    end
    if myprint; fprintf('\n'); end

end

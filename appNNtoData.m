%% Function to apply the NN to a dataset, identify mistakes and arrange 
% the information in two structs for correct classifications and mistakes
% written by ~, last modified July 21, 2019.
%
% Inputs:
    % 1- <Data> is a struct containing a set of inputs (Data.X) and labels (Data.Y)
    % 2- <net> is the trained NN
%
% Outputs:
    % 1- <Lmistake> is a struct that contains information about the mistakes
    % 2- <Lcorrect> is a struct that contains information about the correct classifications
%
%
% struct L may have several fields
% L.raw contains basic information about the data points
    % rows correspond to data points
    % 1st column is the row index of the data point in Data
    % 2nd column is the correct label for data point
    % 3rd column is the wrong label for Lmistake, and 0 for Lcorrect
    % 4th column is the softmax value for the prediction of model

% L.byclass is a cell array that contains similar information as L.raw
%     divided according to class labels

% L.flips may contain the corresponding flip point for each row in L.raw

function [Lmistake,Lcorrect] = appNNtoData(Data,net)
    fprintf('----- processing the data\n');
    tic
    % set up the parameters
    soft = 1; % need the softmax for output
    print_output = 0; % do not need to print the output
    wJac = 0; % do not need the Jacobians\
    nclass = net.nodes(end);

    mcounter = 0; ccounter = 0;
    % set up containers for the processing result
    Lmistake = []; Lmistake.raw = []; %mistakeLF = mistakeL;
    for i = 1:nclass
        Lmistake.byclass{i} = []; Lcorrect.byclass{i} = [];
        Lcorrect.idx{nclass} = []; % indexes corresponding to each class
    end
    % go over the training data
    for i = 1:size(Data.X,1)
        Lcorrect.idx{Data.Y(i)+1}(end+1) = i;
        x = Data.X(i,:);
        mypdata = pointEval(x,net,wJac,print_output,soft);
        [mySoftmaxVal,myclass] = max(mypdata.zs);
        if myclass ~= Data.Y(i)+1
            mcounter = mcounter + 1; % mistake counter
            Lmistake.raw(mcounter,1) = i; % the image index misclassified
            Lmistake.raw(mcounter,2) = Data.Y(i) + 1; % the correct label
            Lmistake.raw(mcounter,3) = myclass; % the wrong label
            Lmistake.raw(mcounter,4) = mySoftmaxVal; % softmax of wrong label
            Lmistake.byclass{Data.Y(i)+1}(end+1,1) = i;
            Lmistake.byclass{Data.Y(i)+1}(end,2) = myclass;
            Lmistake.byclass{Data.Y(i)+1}(end,3) = mySoftmaxVal;
            temp = sort(mypdata.zs);
            Lmistake.byclass{Data.Y(i)+1}(end,4) = temp(end-1);
        else
            ccounter = ccounter + 1; % correct counter
            Lcorrect.raw(ccounter,1) = i; % the image index
            Lcorrect.raw(ccounter,2) = Data.Y(i) + 1; % correct label
            Lcorrect.raw(ccounter,3) = 0; % empty space 
            Lcorrect.raw(ccounter,4) = mySoftmaxVal; % softmax of correct label
    %         Lcorrect.raw(i,:) = [mypdata.zs,Data.Y(i)+1];
            Lcorrect.byclass{Data.Y(i)+1}(end+1,:) = mypdata.zs;
        end
        if mod(i,10000)==0 % print the progress
            fprintf('data id = %d processed!\n', i);
        end
    end
    % Lmistake.writepath = writepath;   Lcorrect.writepath = writepath;
    toc
    fprintf('total number of mistakes found: %d \n',mcounter);
    fprintf('accuracy: %.2f%% \n',100-mcounter/size(Data.X,1)*100);
end

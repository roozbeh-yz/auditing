%% This function reads the parameters of the trained network 
% from text files (in CSV format) and saves the network in a struct named <net>
%
% written by ~, last modified July 21, 2018.
%
% Inputs:
    % 1- <nlayer> is the number of inner layers for the neural net. This is the
        % only required input argument for this function.
    % 2- <foldername> is the name of the folder placed in the current directory
        % that contains the text files. Default value for this parameter is 
        % "trained_net".
    % 3- <currentfolder> is a binary parameter defining whether the text files are
        % in the current path. If <currentfolder> is zero, the full path of the 
        % directory should be passed with the <foldername>.

    % The network structure is feed-forward as described in the readme file.
    % The function finds out the number of nodes based on the entries in the text files.
%
% Output:
    % <net> is the struct containing the information about the trained NN
        % net.nx is the number of elements in the x, input of NN
        % net.nz is the number of output nodes in the NN
        % net.nl is the number of inner layers in the NN
            % first inner layer is the set of nodes immediately after the input layer
            % last inner layer is the set of nodes immediately before the output layer
        % net.nodes is a vector showing the number nodes on each inner layer
        % net.w is a cell array containing the set of weights connecting to each 
            % layer in NN. net.w{i} is the weight matrix connecting to layer i.
        % net.b is a cell array containing the set of bias vectors on each layer in NN. 
            % net.w{i} is the bias vector for layer i.
        % net.sigmaNN is the tuning parameters for the kernels in the network

function [net] = readTrainedNN(nlayer,foldername,currentfolder)
    if nargin < 2; foldername = '/trained_net/'; end
    if nargin < 3; currentfolder = 1; end

    if currentfolder
        currentPath = pwd;
        trainedPath = [currentPath,foldername];
        addpath(trainedPath);
    else
        trainedPath = foldername;
        addpath(trainedPath);
    end

    net.nl = nlayer; % # of inner layers
    net.nodes = []; % number of nodes on each layer

    for i = 1:nlayer+1
        net.w{i} = csvread([trainedPath,'W',num2str(i),'.csv']);  % weights
        net.b{i} = csvread([trainedPath,'B',num2str(i),'.csv'])';  % biases
        net.nodes = [net.nodes,size(net.w{i},2)];
    end
    net.nx = size(net.w{1},1);
    net.sigmaNN = csvread([trainedPath,'sigma.csv'])';  % sigmas 
    net.nz = net.nodes(end);

    fprintf('Read the trained neural net in %s!\n', foldername);

end
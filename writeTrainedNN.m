%% Function to write the parameters of a trained network into text files (CSV format)
%
% written by ~, last modified July 21, 2019.
%
% Inputs:
    % 1- <net> is a struct containing the parameters of the NN
    % 2- <foldername> is the name of folder in which the text files are written -
        % the default value is "trained_net"
    %
    % The written text files are weights (W), biases (B), and the tuning parameters (sigma).
%
% <net> is the struct that contains the information about a trained NN
    % net.nx is the number of elements in the x, input of NN
    % net.nz is the number of output nodes in the NN
    % net.nl is the number of inner layers in the NN
    %       first inner layer is the set of nodes immediately after the input layer
    %       last inner layer is the set of nodes immediately before the output layer
    % net.nodes is a vector showing the number nodes on each inner layer
    % net.w is a cell array containing the set of weights connecting to each 
    %       layer in NN. net.w{i} is the weight matrix connecting to layer i.
    % net.b is a cell array containing the set of bias vectors on each layer in NN. 
    %       net.w{i} is the bias vector for layer i.
    % net.sigmaNN is the tuning parameters for the kernels in the network

function [] = writeTrainedNN(net,foldername)
    if nargin < 2; foldername = '/trained_net/'; end % default directory
    
    currentFolder = pwd;
    trainedPath = [currentFolder,foldername];
    addpath(trainedPath);
    
    for i = 1:nlayer+1
        csvwrite([trainedPath,'W',num2str(i),'.csv'],net.w{i});  % weights
        csvwrite([trainedPath,'B',num2str(i),'.csv'],net.b{i}');  % biases
    end
    
    csvwrite([trainedPath,'sigma.csv'],net.sigmaNN');
    
    fprintf('Wrote the trained neural net in %s!\n', foldername);
end

% Load data
load('facialPoints.mat');
load('headpose.mat');

% process data
labels = pose(:,6);
input = reshape(points,[],8955);
target = transpose(labels);

% Shuffling
shuffle = randperm(8955);
input = input(:,shuffle);
target = target(:,shuffle);

% Parameters
% RxQ1 matrix of Q1 representative R-element input vectors.

P = input;
% SNxQ2 matrix of Q2 representative SN-element target vectors.
T = target;
% Sizes of N-1 hidden layers, S1 to S(N-1), default is [].
S = 30;
% % - Transfer function of ith layer. Default is 'tansig' for hidden layers, and 'purelin' for output layer.
% TF = 'tansig';
% % Backprop network training function, default is 'trainlm'.
% BTF = 'traingdx';
% % Backprop weight/bias learning function, default is 'learngdm'.
% BLF = 'learngdm'; 
% % Performance function, default is 'mse'.
% PF = 'mse';
% % - Row cell array of input processing functions. Default is {'fixunknowns','remconstantrows','mapminmax'}.
% IPF =  {'fixunknowns','remconstantrows','mapminmax'};
% % Row cell array of output processing functions. Default is {'remconstantrows','mapminmax'}.
% OPF = {'remconstantrows','mapminmax'};
% % Data division function, default is 'dividerand';
% DDF = 'dividerand';


% Initialise Network
% net = newff(P,T,S,TF,BTF,BLF,PF,IPF,OPF,DDF);
% net = newff(input,target,10);
% net = newff(P,T,S);

% Alternative
% net = feedforwardnet(10);

% Partitions
part = 10;

% splitting test and train data
[itest,itrain] = kfold(input,part);
[ttest,ttrain] = kfold(target,part);

for a = 1:part
    fprintf('Fold',a);
    %initialise Neural Network     
    net = newff(P,T,S);
    net.trainParam.epochs=1000;
    % training
    [net,tr] = train(net,itrain{a,1},ttrain{a,1});

    % Test the Network
    output = net(itest{a,1});
%     err{a} = gsubtract(ttest{a,1},output);
    performance(a,1) = perform(net,ttest{a,1},output);


end

avg_perf = sum(performance)/10;
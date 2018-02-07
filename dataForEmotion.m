% dataForEmotion
% reformats target data into 1x6 matrix for multiclassEmotion.m

load('emotions_data.mat');

%declare a new matrix for y
newY = zeros(6,612);

%change every data of y into the corresponding matrix
for i = 1:612
  currVal = y(i);
 switch currVal
     case 1 
         newY(:,i) = [1, 0, 0, 0, 0, 0];
     case 2
         newY(:,i) = [0, 1, 0, 0, 0, 0];
     case 3
         newY(:,i) = [0, 0, 1, 0, 0, 0];
     case 4
         newY(:,i) = [0, 0, 0, 1, 0, 0];
     case 5 
         newY(:,i) = [0, 0, 0, 0, 1, 0];
     case 6
         newY(:,i) = [0, 0, 0, 0, 0, 1];
 end
end


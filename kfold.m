function [test, train] = kfold(data, k)

  n = size(data,2);

  test{k,1} = [];
  train{k,1} = [];

  fold = n/k;

  for f = 1:k
      train{f} = data;
      a = round((f-1) * fold + 1, 0);
      z = round( f    * fold    , 0);
      test{f} = data(:,a:z);
      train{f}(:,a:z) = [];
  end
end


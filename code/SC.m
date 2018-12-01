function [U] = SC(fea, options, k)
% 谱聚类的实现
% 输入：
%        -fea: 数据集, n*m的矩阵. 每一列是一个特征, 每一行是一个样例；
%        -options:构建度矩阵的参数；
%        -k: 簇的个数；
% 输出：
%         -U

% 得到度矩阵
W = constructW(fea, options);
degs = sum(W, 2);
D    = sparse(1:size(W, 1), 1:size(W, 2), degs);

% 计算非标准拉普拉斯矩阵
L = D - W;

% 计算标准拉普拉斯矩阵
        % 防止出现其中有元素为0，eps为最小的正数
        degs(degs == 0) = eps;
        % 计算D的逆矩阵
        D = spdiags(1./degs, 0, size(D, 1), size(D, 2));
        % 计算标准拉普拉斯矩阵
        L = D * L;
% 计算U
[U, ~] = eigs(L, k, eps);
end
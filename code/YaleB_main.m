function [a] = YaleB_main()

%% 载入数据库
load YaleB.mat;

%% 构建目标索引
K=gnd';

%% 构建数据集X，其中维数要求为n*d，其中n为数据个数，d为单个数据的特征数量
r=randperm( size(fea,1) );   %生成关于行数的随机排列行数序列
fea=fea(r, :);                              %根据这个序列对A进行重新排序
K=K(r, :);

%% 构建W的相关参数
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 10^4;

%% gamma值
gamma = 1;

%% 调用ShrunkSC函数
U = ShrunkSC( fea, gamma, options );

%% 对G使用K-MEANS聚类
[c,~,~] = kmeans(U,38);

%% 计算ACC
[a] = acc(K,c');

end
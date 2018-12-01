function [a] = JAFFE_main()

%% 载入数据库
load JAFFE.mat;

%% 构建目标索引
K=Y1';

%% 构建数据集X，其中维数要求为n*d，其中n为数据个数，d为单个数据的特征数量
fea = rand(1,676);
for i=1:213
    B = X(i,:);
    BB=reshape(B,64,64);
    A = imresize(BB,[26 26]);
    m=A(:)';
    fea=[fea;m];
end
fea(1,:)=[];
fea = double(fea);

%% 构建W的相关参数
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 10^4;

%% gamma值
gamma = 1;

%% 调用ShrunkSC函数
G = ShrunkSC( fea', gamma, options );

%% 对G使用K-MEANS聚类
[c,~,~] = kmeans(G,10,'Start','plus');

%% 计算ACC
[a] = acc(K,c');

end
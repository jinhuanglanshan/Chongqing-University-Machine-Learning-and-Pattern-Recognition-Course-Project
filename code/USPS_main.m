function [a] = USPS_main()

%% 载入数据库
load USPS.mat;

%% 构建目标索引
I1= ones(1,1100);
K=I1;
for i=2:2
    I= ones(1,1100);
    I11= ones(1,1100);
    for j=1:(i-1)
    I=I+I11;
    end
    K=[K,I];
end

%% 构建数据集X，其中维数要求为n*d，其中n为数据个数，d为单个数据的特征数量
k=data(:,:,:);
fea=rand(256,1);
for i =1:2
    fea=[fea,k(:,:,i)];
end
fea(:,1)=[];
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
G = ShrunkSC( fea, gamma, options );

%% 对G使用K-MEANS聚类
[c,Dsum,z] = kmeans(G,2);

%% 计算ACC
[a] = acc(K,c');
disp(a);

end
function [a] = UMIST_main()

%% 载入数据库
load UMIST.mat;

%% 构建目标索引
I1= ones(1,38);
K=I1;
for i=2:20
    [o,p,q]=size(facedat{1,i});
    I= ones(1,q);
    I11= ones(1,q);
    for j=1:(i-1)
    I=I+I11;
    end
    K=[K,I];
end

%% 构建数据集X，其中维数要求为n*d，其中n为数据个数，d为单个数据的特征数量
fea = rand(1,10304);
for i=1:20
    a=facedat{1,i};
    [b,c,d]=size(a);
    for j=1:d
        k=a(:,:,j);
        f=k';
        m=f(:);
        fea=[fea;m'];
    end
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
[c,Dsum,z] = kmeans(G,20);

%% 计算ACC
[a] = acc(K,c');
disp(a);

%% 将同一类别图片输出到同一文件夹
for ii=1:20
    cc=find(c==ii);
    kk=size(cc);
    for jj=1:kk
        B = fea(cc(jj),:);
        BB=reshape(B,92,112);
        BB=BB';
        k = imshow(uint8(BB));
        I=getimage(k);
        imwrite(I,['./image/',int2str(ii),'/',int2str(jj),'.png']);
    end
end

end
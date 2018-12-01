function [a] = ORL_PCA_SC_main()

%% 载入数据库
load ORL.mat;

%% 构建目标索引
I1= ones(1,10);
K=I1;
for i=2:40
    I= ones(1,10);
    I11= ones(1,10);
    for j=1:(i-1)
    I=I+I11;
    end
    K=[K,I];
end

%% 构建数据集X，其中维数要求为n*d，其中n为数据个数，d为单个数据的特征数量
fea = rand(1,10304);
for i=1:40 %i为拍摄对象的编号
    for j=1:10
    k=filedata{i,j};
    m=k(:)';
    fea=[fea;m];
    end
end
fea(1,:)=[];
fea = double(fea);
r=randperm( size(fea,1) );   %生成关于行数的随机排列行数序列
fea=fea(r, :);                              %根据这个序列对A进行重新排序
K=K';
K=K(r, :);           %对索引进行相同的随机排列

%% 对数据使用PCA降维
[pcaData] = PCA(fea, 100);

%% 对G使用SC聚类
[U] = SC(pcaData, options, 40);
[c,~,~] = kmeans(U,40);

%% 计算ACC
[a] = acc(K,c');
disp(a);

%% 将同一类别图片输出到同一文件夹
for ii=1:40
    cc=find(c==ii);
    kk=size(cc);
    for jj=1:kk
        B = fea(cc(jj),:);
        BB=reshape(B,112,92);
        k = imshow(uint8(BB));
        I=getimage(k);
        imwrite(I,['./image/',int2str(ii),'/',int2str(jj),'.png']);
    end
end

end
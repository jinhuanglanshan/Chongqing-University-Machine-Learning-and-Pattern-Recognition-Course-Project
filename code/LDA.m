function [w] = LDA(data,label,nv)
% 多分类LDA的实现
% 输入：
%        -data: 数据集, n*m的矩阵. 每一列是一个特征, 每一行是一个样例；
%        -label: 数据集的标签, n*1的矩阵, 每一行代表相应样例的标签；
%        -nv: 投影之后的矩阵维度；
% 输出：
%         -w: 投影矩阵；

[m,n]=size(data);
M=mean(data,1); %列均值

Sb=sparse(zeros(n,n));
Sw=sparse(zeros(n,n));

for i=unique(label') %提取所有类别标签
    Xc=data(label==i,:); %当前标签下的所有样本
    [m1,n1]=size(Xc); 
    mec=mean(Xc); %类内均值
    Sw=Sw+(Xc-ones(m1,1)*mec)'*(Xc-ones(m1,1)*mec); %类内散度矩阵
    Sb=Sb+m1*(mec-M)'*(mec-M); %类间散度矩阵
end

[U,V]=eig(Sw\Sb);
B=diag(V);
[B,index]=sort(B,'descend');

w=U(:,index(1:nv,1));
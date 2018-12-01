function [ G ] = ShrunkSC( fea, gamma,c,n,options )
% SSC的实现
% 输入：
%        -fea: 数据集, n*m的矩阵. 每一列是一个特征, 每一行是一个样例；
%        -options:构建度矩阵的参数；
%        -c:分类的个数；
%        -n：数据集的数据量；
%        -gamma：平衡参数；

% 输出：
%         -G;

W = constructW(fea, options);
D = sum(W, 2);%将W按行求和
D = diag(D);%将值放到对角线上
L = D - W;%计算拉普拉斯矩阵
L = full(L);%将拉普拉斯矩阵转换为全矩阵存储
[eigVector, eigValue] = eig(L);%计算拉普拉斯矩阵的特征向量与特征值
tt = diag(eigValue);
[B, IX] = sort(tt, 'ascend');
F = eigVector(:, IX(1:c));%选择前c个特征向量构建spectral embedding

W1 = constructW(F, options);%利用F构建新的相似矩阵
D1 = sum(W1, 2);
D1 = diag(D1);
L1 = D1 - W1;
L1 = full(L1);


t = 1;
G = rand(n, c);%初始化G
while 1%开始收敛
   Ht = G - F;
   St = diag(0.5./sqrt(sum(Ht.*Ht, 2) + eps));%计算S
   G = (St + gamma * L1)\(St*F);%计算G
   
   obj(t) = sum(sqrt(sum((G-F).*(G-F), 2) + eps));
   
   if t > 1
      cver = abs((obj(t) - obj(t - 1)) / obj(t - 1));
      if cver < 10^-3 || t == 30
          break
      end
          
   end

   t = t + 1;
end

end
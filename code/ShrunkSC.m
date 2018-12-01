function [ G ] = ShrunkSC( fea, gamma,c,n,options )
% SSC��ʵ��
% ���룺
%        -fea: ���ݼ�, n*m�ľ���. ÿһ����һ������, ÿһ����һ��������
%        -options:�����Ⱦ���Ĳ�����
%        -c:����ĸ�����
%        -n�����ݼ�����������
%        -gamma��ƽ�������

% �����
%         -G;

W = constructW(fea, options);
D = sum(W, 2);%��W�������
D = diag(D);%��ֵ�ŵ��Խ�����
L = D - W;%����������˹����
L = full(L);%��������˹����ת��Ϊȫ����洢
[eigVector, eigValue] = eig(L);%����������˹�������������������ֵ
tt = diag(eigValue);
[B, IX] = sort(tt, 'ascend');
F = eigVector(:, IX(1:c));%ѡ��ǰc��������������spectral embedding

W1 = constructW(F, options);%����F�����µ����ƾ���
D1 = sum(W1, 2);
D1 = diag(D1);
L1 = D1 - W1;
L1 = full(L1);


t = 1;
G = rand(n, c);%��ʼ��G
while 1%��ʼ����
   Ht = G - F;
   St = diag(0.5./sqrt(sum(Ht.*Ht, 2) + eps));%����S
   G = (St + gamma * L1)\(St*F);%����G
   
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
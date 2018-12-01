function [a] = YaleB_main()

%% �������ݿ�
load YaleB.mat;

%% ����Ŀ������
K=gnd';

%% �������ݼ�X������ά��Ҫ��Ϊn*d������nΪ���ݸ�����dΪ�������ݵ���������
r=randperm( size(fea,1) );   %���ɹ������������������������
fea=fea(r, :);                              %����������ж�A������������
K=K(r, :);

%% ����W����ز���
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 10^4;

%% gammaֵ
gamma = 1;

%% ����ShrunkSC����
U = ShrunkSC( fea, gamma, options );

%% ��Gʹ��K-MEANS����
[c,~,~] = kmeans(U,38);

%% ����ACC
[a] = acc(K,c');

end
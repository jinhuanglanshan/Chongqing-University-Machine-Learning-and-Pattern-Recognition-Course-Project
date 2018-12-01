function [a] = JAFFE_main()

%% �������ݿ�
load JAFFE.mat;

%% ����Ŀ������
K=Y1';

%% �������ݼ�X������ά��Ҫ��Ϊn*d������nΪ���ݸ�����dΪ�������ݵ���������
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

%% ����W����ز���
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 10^4;

%% gammaֵ
gamma = 1;

%% ����ShrunkSC����
G = ShrunkSC( fea', gamma, options );

%% ��Gʹ��K-MEANS����
[c,~,~] = kmeans(G,10,'Start','plus');

%% ����ACC
[a] = acc(K,c');

end
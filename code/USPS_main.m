function [a] = USPS_main()

%% �������ݿ�
load USPS.mat;

%% ����Ŀ������
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

%% �������ݼ�X������ά��Ҫ��Ϊn*d������nΪ���ݸ�����dΪ�������ݵ���������
k=data(:,:,:);
fea=rand(256,1);
for i =1:2
    fea=[fea,k(:,:,i)];
end
fea(:,1)=[];
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
G = ShrunkSC( fea, gamma, options );

%% ��Gʹ��K-MEANS����
[c,Dsum,z] = kmeans(G,2);

%% ����ACC
[a] = acc(K,c');
disp(a);

end
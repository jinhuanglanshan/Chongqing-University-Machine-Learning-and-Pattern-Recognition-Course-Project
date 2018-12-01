function [a] = UMIST_main()

%% �������ݿ�
load UMIST.mat;

%% ����Ŀ������
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

%% �������ݼ�X������ά��Ҫ��Ϊn*d������nΪ���ݸ�����dΪ�������ݵ���������
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
[c,Dsum,z] = kmeans(G,20);

%% ����ACC
[a] = acc(K,c');
disp(a);

%% ��ͬһ���ͼƬ�����ͬһ�ļ���
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
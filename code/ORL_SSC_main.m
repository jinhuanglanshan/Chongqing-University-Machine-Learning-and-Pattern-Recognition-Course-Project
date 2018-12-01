function [a] = ORL_SSC_main()

%% �������ݿ�
load ORL.mat;

%% ����Ŀ������
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
feat = rand(1,10304);
for i=1:40 %iΪ�������ı��
    for j=1:10
    k=filedata{i,j};
    m=k(:)';
    feat=[feat;m];
    end
end
feat(1,:)=[];
feat = double(feat);

%% �������ݼ�X������ά��Ҫ��Ϊn*d������nΪ���ݸ�����dΪ�������ݵ���������
fea = rand(1,1024);
for i=1:40 %iΪ�������ı��
    for j=1:10
    k=filedata{i,j};
    k=imresize(k,[32,32]);
    m=k(:)';
    fea=[fea;m];
    end
end
fea(1,:)=[];
fea = double(fea);
r=randperm(size(fea,1) );   %���ɹ������������������������
fea=fea(r, :);              %����������ж�fea������������
K=K';                       
K=K(r, :);                  %������������ͬ���������
feat=feat(r, :);   

%% ����W����ز���
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 10^4;

%% gammaֵ
gamma = 1;

%% ����ShrunkSC����
G = ShrunkSC( fea, gamma, 40,400, options );


%% ��Gʹ��K-MEANS����
[c,~,~] = kmeans(G,40);

%% ����ACC
[a] = acc(K,c);
disp(a);

%% ��ͬһ���ͼƬ�����ͬһ�ļ���
for ii=1:40
    cc=find(c==ii);
    kk=size(cc);
    for jj=1:kk
        B = feat(cc(jj),:);
        BB=reshape(B,112,92 );
        k = imshow(uint8(BB));
        I=getimage(k);
        imwrite(I,['./image/',int2str(ii),'/',int2str(jj),'.png']);
    end
end

end
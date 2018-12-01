function [a] = ORL_PCA_SC_main()

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

%% �������ݼ�X������ά��Ҫ��Ϊn*d������nΪ���ݸ�����dΪ�������ݵ���������
fea = rand(1,10304);
for i=1:40 %iΪ�������ı��
    for j=1:10
    k=filedata{i,j};
    m=k(:)';
    fea=[fea;m];
    end
end
fea(1,:)=[];
fea = double(fea);
r=randperm( size(fea,1) );   %���ɹ������������������������
fea=fea(r, :);                              %����������ж�A������������
K=K';
K=K(r, :);           %������������ͬ���������

%% ������ʹ��PCA��ά
[pcaData] = PCA(fea, 100);

%% ��Gʹ��SC����
[U] = SC(pcaData, options, 40);
[c,~,~] = kmeans(U,40);

%% ����ACC
[a] = acc(K,c');
disp(a);

%% ��ͬһ���ͼƬ�����ͬһ�ļ���
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
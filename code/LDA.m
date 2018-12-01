function [w] = LDA(data,label,nv)
% �����LDA��ʵ��
% ���룺
%        -data: ���ݼ�, n*m�ľ���. ÿһ����һ������, ÿһ����һ��������
%        -label: ���ݼ��ı�ǩ, n*1�ľ���, ÿһ�д�����Ӧ�����ı�ǩ��
%        -nv: ͶӰ֮��ľ���ά�ȣ�
% �����
%         -w: ͶӰ����

[m,n]=size(data);
M=mean(data,1); %�о�ֵ

Sb=sparse(zeros(n,n));
Sw=sparse(zeros(n,n));

for i=unique(label') %��ȡ��������ǩ
    Xc=data(label==i,:); %��ǰ��ǩ�µ���������
    [m1,n1]=size(Xc); 
    mec=mean(Xc); %���ھ�ֵ
    Sw=Sw+(Xc-ones(m1,1)*mec)'*(Xc-ones(m1,1)*mec); %����ɢ�Ⱦ���
    Sb=Sb+m1*(mec-M)'*(mec-M); %���ɢ�Ⱦ���
end

[U,V]=eig(Sw\Sb);
B=diag(V);
[B,index]=sort(B,'descend');

w=U(:,index(1:nv,1));
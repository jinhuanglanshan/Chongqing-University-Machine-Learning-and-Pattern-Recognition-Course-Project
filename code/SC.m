function [U] = SC(fea, options, k)
% �׾����ʵ��
% ���룺
%        -fea: ���ݼ�, n*m�ľ���. ÿһ����һ������, ÿһ����һ��������
%        -options:�����Ⱦ���Ĳ�����
%        -k: �صĸ�����
% �����
%         -U

% �õ��Ⱦ���
W = constructW(fea, options);
degs = sum(W, 2);
D    = sparse(1:size(W, 1), 1:size(W, 2), degs);

% ����Ǳ�׼������˹����
L = D - W;

% �����׼������˹����
        % ��ֹ����������Ԫ��Ϊ0��epsΪ��С������
        degs(degs == 0) = eps;
        % ����D�������
        D = spdiags(1./degs, 0, size(D, 1), size(D, 2));
        % �����׼������˹����
        L = D * L;
% ����U
[U, ~] = eigs(L, k, eps);
end
function [pcaData,projectionVectors,eigVal] = PCA( data, featuresToExtract)
    
% PCA��ʵ��
% ���룺
%        -data: ���ݼ�, n*m�ľ���. ÿһ����һ������, ÿһ����һ��������
%        -label: ���ݼ��ı�ǩ, n*1�ľ���, ÿһ�д�����Ӧ�����ı�ǩ��
%        -nv: ͶӰ֮��ľ���ά�ȣ�
% �����
%         -w: ͶӰ����

%
% myPCA( data, numberOfFeatures )
%
% ���룺 data: ���ݼ�, n*m�ľ���. ÿһ����һ������, ÿһ����һ��������
%            featuresToExtract: ��Ҫ����������ȡ������ֵ��������
%
% ����� pcaData: PCA���������ݼ� 
%        eigVec: ��PCA�еõ�����������
%        eigVal: ��PCA�еõ�������ֵ
%
    data = double(data);
    % �õ����ݼ�����������������ֵ����
    [numberOfExamples,numberOfFeatures] = size(data);
    
    % ��׼������
    
    % �õ����ݼ���ÿһ�еľ�ֵ
    dataMean = mean(data,1);
    
    % ��ʼ��
    normalizedData = zeros(numberOfExamples,numberOfFeatures);
    
    for i = 1 : numberOfExamples
        normalizedData(i,:) = data(i,:) - dataMean;%���Ļ�
    end
    
    
    % ����Э�������
    covarianceMatrix = cov(normalizedData);
    
    % ����ֵ�ֽ�
    [eigVec, eigVal] = eig(covarianceMatrix);
    eigVal = diag(eigVal);
    bestEigVal = sortrows(eigVal,-1);
    
    %��ά����ȡǰfeaturesToExtract����������
    for i = 1 : featuresToExtract 
        projectionVectors(:,i) = eigVec(:,eigVal == bestEigVal(i));
    end
    
    eigVal = bestEigVal;
    %�õ��µ����ݼ�
    pcaData = normalizedData * projectionVectors;
    
end
   
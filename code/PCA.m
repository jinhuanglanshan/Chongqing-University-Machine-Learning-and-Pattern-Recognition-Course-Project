function [pcaData,projectionVectors,eigVal] = PCA( data, featuresToExtract)
    
% PCA的实现
% 输入：
%        -data: 数据集, n*m的矩阵. 每一列是一个特征, 每一行是一个样例；
%        -label: 数据集的标签, n*1的矩阵, 每一行代表相应样例的标签；
%        -nv: 投影之后的矩阵维度；
% 输出：
%         -w: 投影矩阵；

%
% myPCA( data, numberOfFeatures )
%
% 输入： data: 数据集, n*m的矩阵. 每一列是一个特征, 每一行是一个样例；
%            featuresToExtract: 需要从数据中提取的特征值的数量；
%
% 输出： pcaData: PCA处理后的数据集 
%        eigVec: 从PCA中得到的特征向量
%        eigVal: 从PCA中得到的特征值
%
    data = double(data);
    % 得到数据集的样例个数和特征值个数
    [numberOfExamples,numberOfFeatures] = size(data);
    
    % 标准化数据
    
    % 得到数据集中每一列的均值
    dataMean = mean(data,1);
    
    % 初始化
    normalizedData = zeros(numberOfExamples,numberOfFeatures);
    
    for i = 1 : numberOfExamples
        normalizedData(i,:) = data(i,:) - dataMean;%中心化
    end
    
    
    % 计算协方差矩阵
    covarianceMatrix = cov(normalizedData);
    
    % 特征值分解
    [eigVec, eigVal] = eig(covarianceMatrix);
    eigVal = diag(eigVal);
    bestEigVal = sortrows(eigVal,-1);
    
    %降维，提取前featuresToExtract个特征向量
    for i = 1 : featuresToExtract 
        projectionVectors(:,i) = eigVec(:,eigVal == bestEigVal(i));
    end
    
    eigVal = bestEigVal;
    %得到新的数据集
    pcaData = normalizedData * projectionVectors;
    
end
   
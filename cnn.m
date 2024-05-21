clc
clear
close all
% image_path = 'C:\Users\szq\Desktop\cnn\';
image_path = fullfile('C:','Users/szq/Desktop/cnn/data/');
% 创建ImageDatastore对象
imds = imageDatastore( image_path,'IncludeSubfolders',true,"LabelSource",'foldernames');

[imdsTrain,imdsVal]=splitEachLabel(imds,0.8,0.2,'randomized');
% imdsTrain.ReadFcn = @(filename)imresize(imread(filename), [32 32]); % 将图像大小调整为 32x32
% imdsVal.ReadFcn = @(filename)imresize(imread(filename), [32 32]); % 将图像大小调整为 32x32
%创建卷积神经网络模型
layers = [
    imageInputLayer([256 256 3])    % 输入层
    batchNormalizationLayer         % 批量归一化层
    convolution2dLayer(4,40)        % 卷积层，5x5卷积核，20个卷积核
    batchNormalizationLayer(); 
    reluLayer                       % ReLU激活层
    maxPooling2dLayer(2,'Stride',2) % 最大池化层，2x2池化核，步长为2

    convolution2dLayer(8,80)        % 卷积层，5x5卷积核，40个卷积核
    batchNormalizationLayer         % 批量归一化层
    reluLayer                       % ReLU激活层
    maxPooling2dLayer(4,'Stride',4) % 最大池化层，2x2池化核，步长为2

    fullyConnectedLayer(2)          % 全连接层，输出类别数为2
    batchNormalizationLayer         % 批量归一化层
    reluLayer                       % ReLU激活层
    fullyConnectedLayer(2)          % 全连接层，输出类别数为2
    softmaxLayer                    % Softmax激活层
    classificationLayer             % 分类层    
    ];
%设置训练选项
options = trainingOptions('adam',...
                        'MaxEpochs',100,...
                        'InitialLearnRate', 0.001, ...
                        'ValidationData',imdsVal,...
                        'GradientThreshold',1,...
                        'ValidationFrequency',10,...
                        'Verbose',true, ...
                        'Plots','training-progress',...
                        'ValidationPatience',inf, ...
                        'MiniBatchSize',20, ...
                        'Shuffle','every-epoch'); 

%训练卷积神经网络模型
net = trainNetwork(imdsTrain,layers,options);
 %在测试集上评估模型性能
Pre = classify(net,imdsVal);
Vaildation = imdsVal.Labels;
accuracy = mean(Pre==Vaildation);
# Multimodal-analysis-of-infant-crying
环境:
python 3.10.15
torch 1.13.1
tensorflow 2.10.0

数据处理方面采取了音频增强策略，并且对于音频进行分割，将音频文件分割为10s的数据（不足10s的数据将会被扩充到10s）。采取了STFT方式将音频转化为频谱图的方式送入模型中进行计算。在数据处理的过程中将会保留分类好的标签，使得模型按照既定的标签进行分类音频数据。

模型部分使用CNN+DNN搭建网络，
CNN部分使用了四个卷积层，Dropout层随机丢弃50%神经元，DNN部分有两个全连接层最后输出层输出层输出维度是n_classes，即分类任务中的类别数，激活函数使用ReLU激活函数，并且使用CrossEntropyLoss作为损失函数。

优化器选择：Adam

    # 配置如下
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,            # 学习率
        betas=(0.9, 0.999),  # beta1, beta2
        eps=1e-7,            # epsilon
        weight_decay=1e-5,   # L2正则化
        amsgrad = False      # 是否使用AMSGrad
    )

训练过程中使用了CosineAnnealingLR学习率调度器，采用了早期停止策略，通过训练结果绘制训练和验证过程中的损失曲线、准确率曲线、学习率曲线。在测试集评估中通过混淆矩阵和准确率来进行模型评估

最后保存模型和标签编码
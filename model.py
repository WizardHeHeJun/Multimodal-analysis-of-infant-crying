import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义声音分类模型
class SoundModel(nn.Module):
    # 初始化模型
    def __init__(self, n_classes):
        super(SoundModel, self).__init__()
        # 第一个卷积层，输入2通道（STFT和Mel）
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 第三个卷积层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # 第四个卷积层，增加深度
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)

        # 全连接层
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)


    # 前向传播
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # 展平
        x = x.view(-1, 256 * 8 * 8)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
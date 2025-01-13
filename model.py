import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import device

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SoundModel(nn.Module):
    def __init__(self, n_classes):
        super(SoundModel, self).__init__()

        # CNN部分
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 批标准化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 批标准化
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # 批标准化
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  # 批标准化

        # Max Pooling层
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # DNN部分（全连接层）
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # 将卷积层输出展平后输入到全连接层
        self.fc2 = nn.Linear(512, 256)  # 第二个全连接层，256个神经元
        self.fc3 = nn.Linear(256, n_classes)  # 输出层，n_classes是类别数

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # CNN部分
        x = self.pool(self.relu(self.conv1(x)))  # 第一层卷积 + 池化
        x = self.pool(self.relu(self.conv2(x)))  # 第二层卷积 + 池化
        x = self.pool(self.relu(self.conv3(x)))  # 第三层卷积 + 池化
        x = self.pool(self.relu(self.conv4(x)))  # 第四层卷积 + 池化

        # 展平
        x = x.view(-1, 256 * 8 * 8)

        # DNN部分
        x = self.relu(self.fc1(x))  # 第一个全连接层
        x = self.dropout(x)  # Dropout
        x = self.relu(self.fc2(x))  # 第二个全连接层
        x = self.fc3(x)  # 输出层

        return x

# 初始化模型并将其移到GPU
def initialize_model(n_classes):
    model = SoundModel(n_classes=n_classes).to(device)
    return model

# 定义损失函数和优化器
def get_optimizer_and_criterion(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # 正则化
    return optimizer, criterion

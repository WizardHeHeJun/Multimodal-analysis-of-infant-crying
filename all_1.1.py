import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.cuda import device
from torch.utils.data import DataLoader, Dataset
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 设置常量
image_size = (128, 128)  # 输入频谱图的尺寸
n_classes = 7  # 设定类别数

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 音频文件转频谱图的函数
def audio_to_spectrogram(audio_path, size=(128, 128), duration=5, augment=False):
    # 加载音频文件，指定持续时间
    y, sr = librosa.load(audio_path, sr=None, duration=duration)

    if augment:
        noise = np.random.randn(len(y)) * 0.005
        y = y + noise

    # 如果音频长度小于5秒，填充到5秒
    if len(y) < sr * duration:
        pad_length = sr * duration - len(y)
        y = np.pad(y, (0, pad_length), mode='constant')

    # 计算频谱图
    D = librosa.stft(y)
    D_mag, _ = librosa.magphase(D)
    S_db = librosa.amplitude_to_db(D_mag, ref=np.max)

    # 裁剪或填充频谱图
    if S_db.shape[1] > size[1]:
        S_db_resized = S_db[:, :size[1]]  # 如果宽度大于目标尺寸，裁剪
    else:
        pad_width = size[1] - S_db.shape[1]
        S_db_resized = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')  # 填充宽度

    # 固定高度
    if S_db_resized.shape[0] > size[0]:
        S_db_resized = S_db_resized[:size[0], :]
    else:
        pad_height = size[0] - S_db_resized.shape[0]
        S_db_resized = np.pad(S_db_resized, ((0, pad_height), (0, 0)), mode='constant')  # 填充高度

    return S_db_resized

# 数据加载和标签生成
class AudioDataset(Dataset):
    def __init__(self, data_dir, image_size=(128, 128), duration=5, augment=False):
        self.X = []
        self.y = []
        self.image_size = image_size
        self.duration = duration
        self.label_encoder = LabelEncoder()  # 用于标签编码
        self.augment = augment  # 是否启用数据增强

        # 遍历数据目录，加载音频文件并转换为频谱图
        for label in os.listdir(data_dir):
            class_folder = os.path.join(data_dir, label)
            if os.path.isdir(class_folder):
                for file in os.listdir(class_folder):
                    if file.endswith('.wav'):
                        file_path = os.path.join(class_folder, file)
                        spec = audio_to_spectrogram(file_path, size=self.image_size, duration=self.duration, augment=self.augment)
                        self.X.append(spec)
                        self.y.append(label)

        # 标签编码
        self.y = self.label_encoder.fit_transform(self.y)

        # 转换为numpy数组
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        # 扩展X的维度以适配模型输入
        self.X = np.expand_dims(self.X, axis=-1)
        self.X = np.transpose(self.X, (0, 3, 1, 2))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32).to(device), torch.tensor(self.y[idx],dtype=torch.long).to(device)

# 加载数据
data_dir = 'data'
dataset = AudioDataset(data_dir, image_size=image_size, duration=5)

# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#模型构建
class CryingSoundModel(nn.Module):
    def __init__(self, n_classes=7):
        super(CryingSoundModel, self).__init__()

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
        x = self.pool(self.relu(self.conv4(x)))

        # 展平
        x = x.view(-1, 256 * 8 * 8)

        # DNN部分
        x = self.relu(self.fc1(x))  # 第一个全连接层
        x = self.dropout(x)  # Dropout
        x = self.relu(self.fc2(x))  # 第二个全连接层
        x = self.fc3(x)  # 输出层

        return x

# 初始化模型并将其移到GPU
model = CryingSoundModel(n_classes=n_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练模型
num_epochs = 100
train_losses = []
train_accuracies = []
val_losses = []

# 早期停止
best_loss = float('inf')
patience = 10  # 容忍训练轮次
counter = 0

#ReduceLROnPlateau学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        running_loss += loss.item()

    # 更新学习率
    scheduler.step(running_loss)  # 使用loss来控制学习率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_preds / total_preds
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # 早期停止策略
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        print("Early stopping triggered!")
        break

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

# 绘制训练过程中的损失和准确率曲线
plt.figure(figsize=(12, 6))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.show()

# 获取预测结果
all_preds = []
all_labels = []

# 模型评估
model.eval()  # 设置为评估模式
running_loss = 0.0
correct_preds = 0
total_preds = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        running_loss += loss.item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.label_encoder.classes_, yticklabels=dataset.label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#计算测试集准确率
accuracy = correct_preds / total_preds
print(f"Test Accuracy: {accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), 'crying_sound_model.pth')

# 加载模型
model = CryingSoundModel(n_classes=n_classes).to(device)
model.load_state_dict(torch.load('crying_sound_model.pth'))
model.eval()

# 获取测试样本并进行预测
sample, _ = test_dataset[0]  # 从测试集获取一个样本
sample = sample.clone().detach().unsqueeze(0).float()# 添加batch维度
sample = sample.to(device)  # 确保使用GPU进行预测

# 进行预测
model.eval()
with torch.no_grad():
    prediction = model(sample)
    _, predicted_class = torch.max(prediction, 1)

predicted_label = dataset.label_encoder.inverse_transform([predicted_class.item()])
print(f"Predicted label: {predicted_label[0]}")
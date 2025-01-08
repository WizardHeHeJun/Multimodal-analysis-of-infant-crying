import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import device
from train import CryingSoundModel, n_classes, test_dataset, dataset

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
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
from model import SoundModel

# 设置常量
n_classes = 5  # 设定类别数
num_epochs = 200  # 训练轮次
data_dir = 'data'  # 数据加载地址
n_classes = 5  # 设定类别数
image_size = (128, 128)  # 输入频谱图的尺寸
duration = 10  # 设定裁剪时长

def audio_to_spectrogram(audio_path, size, duration, augment=False):
    # 加载音频文件，指定持续时间
    y, sr = librosa.load(audio_path, sr=None, duration=duration)

    if augment:
        noise = np.random.randn(len(y)) * 0.005
        y = y + noise

    # 如果音频长度小于10秒，填充到10秒
    if len(y) < sr * duration:
        pad_length = sr * duration - len(y)
        y = np.pad(y, (0, pad_length), mode='constant')

    # 设置 n_fft 和 hop_length 参数
    n_fft = 1024  # 设置为 1024，表示每个 STFT 窗口的大小为 1024
    hop_length = n_fft // 4  # 每帧之间的步长为 n_fft 的四分之一
    win_length = n_fft  # 窗口长度与 n_fft 相同
    window = 'hann'  # 使用汉宁窗

    # 计算 STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
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


def predict_from_audio(audio_path, model, device, image_size, duration):
    # 使用音频处理函数生成频谱图
    spec = audio_to_spectrogram(audio_path, size=image_size, duration=duration)

    # 添加batch维度并转换为tensor
    spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 加上 batch 和 channel 维度
    spec = spec.to(device)  # 移动到GPU（如果使用）

    # 进行预测
    model.eval()
    with torch.no_grad():
        prediction = model(spec)
        _, predicted_class = torch.max(prediction, 1)

    return predicted_class.item()


# 加载模型
model = SoundModel(n_classes=n_classes).to(device)
model.load_state_dict(torch.load('crying_sound_model.pth'))
model.eval()

# 音频文件路径
audio_path = 'path_to_audio_file.wav'

# 进行预测
predicted_class = predict_from_audio(audio_path, model, device, image_size, duration)

# 获取预测的标签
predicted_label = test_dataset.dataset.label_encoder.inverse_transform([predicted_class])
print(f"Predicted label: {predicted_label[0]}")
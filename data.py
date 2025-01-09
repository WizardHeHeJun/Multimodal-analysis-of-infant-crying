import os
import librosa
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.cuda import device

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 音频文件转频谱图
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
        S_db_resized = S_db[:, :size[1]] # 如果宽度大于目标尺寸，裁剪
    else:
        pad_width = size[1] - S_db.shape[1]
        S_db_resized = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant') # 填充宽度

    # 固定高度
    if S_db_resized.shape[0] > size[0]:
        S_db_resized = S_db_resized[:size[0], :]
    else:
        pad_height = size[0] - S_db_resized.shape[0]
        S_db_resized = np.pad(S_db_resized, ((0, pad_height), (0, 0)), mode='constant') # 填充高度

    return S_db_resized

# 数据加载和标签生成
class AudioDataset:
    def __init__(self, data_dir, augment=False):
        self.X = []
        self.y = []
        self.image_size = (128, 128) # 输入频谱图的尺寸
        self.duration = 10 # 设定裁剪时长
        self.label_encoder = LabelEncoder() # 用于标签编码
        self.augment = augment # 是否启用数据增强

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
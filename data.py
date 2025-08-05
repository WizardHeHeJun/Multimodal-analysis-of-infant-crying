import librosa
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder


# 音频转频谱图函数
def audio_to_spectrogram(audio_path, duration=10, size=(128, 128), augment=False):
    # 加载音频
    try:
        y, sr = librosa.load(audio_path, duration=duration)
    except Exception as e:
        print(f"无法加载音频文件 {audio_path}: {e}")
        return None

    # 添加噪声增强
    if augment:
        noise = np.random.randn(len(y)) * 0.005
        y += noise

    # 填充音频
    if len(y) < sr * duration:
        y = np.pad(y, (0, sr * duration - len(y)), mode='constant')

    # 计算STFT
    D = librosa.stft(y, n_fft=2048, hop_length=512,window = 'hann') # 窗口大小2048，会获得1025个频率维度
                                                                    # 每帧之间的步长为512
                                                                    # 特征矩阵形状为(1025,时间帧数)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # 计算Mel频谱图
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128) # FFT窗口的大小2048
                                                                                           # 每帧之间的步长为512
                                                                                           # 梅尔频率的数量为128，梅尔频谱图的频率维度是128
                                                                                           # 特征矩阵形状为(128,时间帧数)
    S_db_mel = librosa.power_to_db(S, ref=np.max)

    # 调整STFT尺寸
    S_db = S_db[:, :size[1]] if S_db.shape[1] > size[1] else np.pad(S_db, ((0, 0), (0, size[1] - S_db.shape[1])),
                                                                    mode='constant')
    S_db = S_db[:size[0], :] if S_db.shape[0] > size[0] else np.pad(S_db, ((0, size[0] - S_db.shape[0]), (0, 0)),
                                                                    mode='constant')

    # 调整Mel尺寸
    S_db_mel = S_db_mel[:, :size[1]] if S_db_mel.shape[1] > size[1] else np.pad(S_db_mel, (
    (0, 0), (0, size[1] - S_db_mel.shape[1])), mode='constant')
    S_db_mel = S_db_mel[:size[0], :] if S_db_mel.shape[0] > size[0] else np.pad(S_db_mel, (
    (0, size[0] - S_db_mel.shape[0]), (0, 0)), mode='constant')

    # 堆叠STFT和Mel
    spectrogram = np.stack((S_db, S_db_mel), axis=0) # 堆叠为一个形状为 (2, target_freq_bins, target_time_frames) 的三维矩阵
                                                           # 第一个维度 2 表示 STFT 和 Mel 频谱图两个特征；
                                                           # 第二个维度 target_freq_bins 是频率维度；
                                                           # 第三个维度 target_time_frames 是时间维度。
    return spectrogram


# 音频数据集类
class AudioDataset(Dataset):
    def __init__(self, spectrograms_dir, augment=False):
        self.spectrograms_dir = spectrograms_dir
        self.augment = augment
        self.data = []
        self.labels = []
        self.label_encoder = LabelEncoder()

        # 验证目录是否存在
        if not os.path.exists(spectrograms_dir):
            raise ValueError(f"数据目录 {spectrograms_dir} 不存在")

        print(f"正在从 {spectrograms_dir} 加载数据...")
        # 遍历目录
        for folder in os.listdir(spectrograms_dir):
            folder_path = os.path.join(spectrograms_dir, folder)
            if os.path.isdir(folder_path):
                file_count = 0
                for file in os.listdir(folder_path):
                    if file.lower().endswith('.wav'):  # 支持 .wav 和 .WAV
                        file_path = os.path.join(folder_path, file)
                        self.data.append(file_path)
                        self.labels.append(folder)
                        file_count += 1
                print(f"类别 {folder}: 加载 {file_count} 个音频文件")

        if not self.data:
            raise ValueError(f"未在 {spectrograms_dir} 中找到任何 .wav 文件")

        # 标签编码
        self.labels = self.label_encoder.fit_transform(self.labels)
        print(f"总计加载 {len(self.data)} 个样本，{len(set(self.labels))} 个类别")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data[idx]
        label = self.labels[idx]
        spectrogram = audio_to_spectrogram(audio_path, augment=self.augment)
        if spectrogram is None:
            raise ValueError(f"无法生成 {audio_path} 的频谱图")
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return spectrogram, label
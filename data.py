import os
import librosa
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.cuda import device

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def audio_to_spectrogram(audio_path, size=(128, 128), duration=10, augment=False):
    # 音频转频谱图的功能保持不变
    y, sr = librosa.load(audio_path, sr=None, duration=duration)

    if augment:
        noise = np.random.randn(len(y)) * 0.005
        y = y + noise

    if len(y) < sr * duration:
        pad_length = sr * duration - len(y)
        y = np.pad(y, (0, pad_length), mode='constant')

    n_fft = 1024
    hop_length = n_fft // 4
    win_length = n_fft
    window = 'hann'

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    D_mag, _ = librosa.magphase(D)
    S_db = librosa.amplitude_to_db(D_mag, ref=np.max)

    if S_db.shape[1] > size[1]:
        S_db_resized = S_db[:, :size[1]]
    else:
        pad_width = size[1] - S_db.shape[1]
        S_db_resized = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')

    if S_db_resized.shape[0] > size[0]:
        S_db_resized = S_db_resized[:size[0], :]
    else:
        pad_height = size[0] - S_db_resized.shape[0]
        S_db_resized = np.pad(S_db_resized, ((0, pad_height), (0, 0)), mode='constant')

    return S_db_resized

class AudioDataset:
    def __init__(self, data_dir, image_size=(128, 128), duration=5, augment=False):
        self.X = []
        self.y = []
        self.image_size = image_size
        self.duration = duration
        self.label_encoder = LabelEncoder()
        self.augment = augment

        for label in os.listdir(data_dir):
            class_folder = os.path.join(data_dir, label)
            if os.path.isdir(class_folder):
                for file in os.listdir(class_folder):
                    if file.endswith('.wav'):
                        file_path = os.path.join(class_folder, file)
                        spec = audio_to_spectrogram(file_path, size=self.image_size, duration=self.duration, augment=self.augment)
                        self.X.append(spec)
                        self.y.append(label)

        self.y = self.label_encoder.fit_transform(self.y)
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        self.X = np.expand_dims(self.X, axis=-1)
        self.X = np.transpose(self.X, (0, 3, 1, 2))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32).to(device), torch.tensor(self.y[idx],
                                                                                       dtype=torch.long).to(device)

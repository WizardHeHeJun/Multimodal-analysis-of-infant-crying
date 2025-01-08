import librosa
import numpy as np

data_dir = 'data'

def audio_to_spectrogram(audio_path, size=(128, 128), duration=10, augment=False):
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

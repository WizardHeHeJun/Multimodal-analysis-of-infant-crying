import os
import joblib
import torch
from data import audio_to_spectrogram
from model import SoundModel

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_from_audio(audio_path, model, device, image_size, duration , LABEL_ENCODER ):

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

    # 加载标签编码器
    label_encoder = joblib.load(LABEL_ENCODER)

    # 解码预测结果
    predicted_label = label_encoder.inverse_transform(predicted_class.cpu().numpy())
    return predicted_label[0]

def predict_directory(audio_dir,model, device,image_size, duration , LABEL_ENCODER ):

    predictions = {}

    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):  # 只处理 .wav 文件
                audio_path = os.path.join(root, file)
                predicted_label = predict_from_audio(audio_path, model , device, image_size , duration , LABEL_ENCODER )
                predictions[audio_path] = predicted_label

    return predictions

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 设置常量
    n_classes = 5  # 设定类别数
    image_size = (128, 128)  # 输入频谱图的尺寸
    duration = 10  # 设定裁剪时长

    #设置模型和标签
    MODEL='sound_model.pth'
    LABEL_ENCODER='label_encoder.pkl'

    # # 音频文件路径
    # audio_path = 'E:\Multimodal-analysis-of-infant-crying\data\hungry\\6A7KZR1p.wav'

    # 音频目录路径
    audio_dir = 'E:\Multimodal-analysis-of-infant-crying\data\hungry'

    # 加载模型
    model = SoundModel(n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(MODEL))
    model.eval()

    # # 对音频文件进行预测
    # predicted_class = predict_from_audio(audio_path, model, device, image_size, duration)
    # print(f"Predicted label: {predicted_class}")

    # 对目录音频进行预测
    predictions = predict_directory( audio_dir , model, device , image_size , duration , LABEL_ENCODER )
    print(f"预测结果：")
    for audio_path, predicted_label in predictions.items():
        print(f"{audio_path} -> {predicted_label}")

if __name__ == "__main__":
    main()

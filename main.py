import torch
from train import train_and_evaluate
import os

# 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置参数
    num_epochs = 100
    n_classes = 8
    # 使用绝对路径，确保正确指向数据目录
    base_dir = r"E:\Multimodal-analysis-of-infant-crying"
    spectrograms_dir = os.path.join(base_dir, "train1")
    if not os.path.exists(spectrograms_dir):
        raise ValueError(f"数据目录 {spectrograms_dir} 不存在")

    # 训练和评估
    model, accuracy = train_and_evaluate(num_epochs, spectrograms_dir, n_classes)
    print(f"训练完成，最终准确率: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
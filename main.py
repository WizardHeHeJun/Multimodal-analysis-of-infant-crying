import torch
from train import train_and_evaluate

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 设置数据
    num_epochs = 300 #训练轮次
    n_classes = 10 # 设定类别数
    data_dir = 'urbansound8k' #数据加载地址

    # 开始训练
    train_and_evaluate(num_epochs, data_dir, n_classes)

if __name__ == "__main__":
    main()
import torch
from train import train_and_evaluate
from model import initialize_model
from data import AudioDataset

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 200 #训练轮次
    patience = 10 #容忍训练轮次

    # 开始训练
    train_and_evaluate(num_epochs, patience)

if __name__ == "__main__":
    main()

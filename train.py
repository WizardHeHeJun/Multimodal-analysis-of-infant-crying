import torch
from torch.utils.data import random_split, DataLoader
from data import AudioDataset
from model import SoundModel
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import joblib


# 训练和评估函数
def train_and_evaluate(num_epochs, spectrograms_dir, n_classes):
    # 加载并分割数据集
    dataset = AudioDataset(spectrograms_dir=spectrograms_dir, augment=True)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 设置设备并初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoundModel(n_classes).to(device)
    # 使用CrossEntropyLoss作为损失函数
    criterion = nn.CrossEntropyLoss()
    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    # 使用CosineAnnealingLR学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    # 初始化指标记录
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []

    # 早停参数
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    # 训练和验证循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # 计算训练指标
        train_accuracy = 100 * correct_train / total_train
        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # 计算验证指标
        val_accuracy = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(test_loader)

        # 记录指标
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # 打印结果
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
        print(f'验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_urbansound8k_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

        # 更新学习率
        scheduler.step()

    # 最终测试
    model.eval()
    final_correct = 0
    final_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            final_total += labels.size(0)
            final_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_accuracy = 100 * final_correct / final_total
    print(f'最终测试准确率: {final_accuracy:.2f}%')

    # 保存最终模型和标签编码器
    torch.save(model.state_dict(), 'urbansound8k_model_final.pth')
    joblib.dump(dataset.label_encoder, 'label_encoder.pkl')
    print("保存最终模型和标签编码器")

    # 绘制损失曲线和准确率曲线
    plt.figure(figsize=(18, 6)) # 画布大小

    plt.subplot(1, 3, 1) # 1行3列，位置为第1个
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curves')
    plt.legend()
    plt.savefig('loss_curve.png')

    plt.subplot(1, 3, 2) # 1行3列，位置为第2个
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy curves')
    plt.legend()
    plt.savefig('accuracy_curve.png')

    # 绘制学习率曲线
    plt.subplot(1, 3, 3) # 1行3列，位置为第3个
    plt.plot(range(1, len(learning_rates) + 1), learning_rates, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Time')
    plt.legend()
    plt.savefig('lr_curve.png')

    plt.show()

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns. heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=dataset.label_encoder.classes_,
                yticklabels=dataset.label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    return model, final_accuracy
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.cuda import device
from torch.utils.data import DataLoader
from model import initialize_model, get_optimizer_and_criterion
from data import AudioDataset
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_evaluate(num_epochs, data_dir, n_classes):

    dataset = AudioDataset(data_dir)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 设置训练批次大小
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型和优化器
    model = initialize_model(n_classes)
    optimizer, criterion = get_optimizer_and_criterion(model)

    # model = model.to(device)

    # 早期停止
    patience = 15 #容忍训练轮次
    best_loss = float('inf')
    counter = 0

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

    #ReduceLROnPlateau学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.001, verbose=True)

    # # 使用CosineAnnealingLR学习率调度器
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            running_loss += loss.item()

        # 更新学习率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # 验证集评估
        model.eval()  # 切换到评估模式
        val_running_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0

        with torch.no_grad(): # 不计算梯度
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                val_total_preds += labels.size(0)
                val_correct_preds += (predicted == labels).sum().item()

                val_running_loss += loss.item()

        # 验证损失和准确率
        val_loss = val_running_loss / len(test_loader)
        val_acc = val_correct_preds / val_total_preds
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # 学习率调度
        scheduler.step(val_loss)

        # 早期停止策略
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

    # 绘制训练和验证过程中的损失和准确率曲线
    plt.figure(figsize=(12, 6))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

    # 测试集评估
    model.eval()
    test_correct_preds = 0
    test_total_preds = 0

    # 获取预测结果
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            test_total_preds += labels.size(0)
            test_correct_preds += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.label_encoder.classes_,
                yticklabels=dataset.label_encoder.classes_)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # 计算测试集准确率
    test_accuracy = test_correct_preds / test_total_preds
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'urbansound8k_sound_model_1.pth')

    # 保存标签编码器
    joblib.dump(dataset.label_encoder, 'urbansound8k_label_encoder.pkl')

    # # 加载模型
    # model = initialize_model(n_classes)
    # model.load_state_dict(torch.load('urbansound8k_sound_model_1.pth'))
    # model.eval()
    #
    # # 获取测试样本并进行预测
    # sample, _ = test_dataset[0]  # 从测试集获取一个样本
    # sample = sample.clone().detach().unsqueeze(0).float()  # 添加batch维度
    # sample = sample.to(device)  # 确保使用GPU进行预测
    #
    # # 进行预测
    # model.eval()
    # with torch.no_grad():
    #     prediction = model(sample)
    #     _, predicted_class = torch.max(prediction, 1)
    #
    # predicted_label = dataset.label_encoder.inverse_transform([predicted_class.item()])
    # print(f"Predicted label: {predicted_label[0]}")

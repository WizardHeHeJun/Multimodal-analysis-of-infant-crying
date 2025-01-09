import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.cuda import device
from torch.utils.data import DataLoader
from model import initialize_model, get_optimizer_and_criterion
from data import AudioDataset

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_evaluate(num_epochs, patience):
    # 设置数据和加载器
    data_dir = 'data'
    dataset = AudioDataset(data_dir, image_size=(128, 128), duration=5)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型和优化器
    model = initialize_model(n_classes=5)  # n_classes根据需要调整
    optimizer, criterion = get_optimizer_and_criterion(model)

    model = model.to(device)

    best_loss = float('inf')
    counter = 0

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

    # 使用CosineAnnealingLR调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # 验证
        model.eval()
        val_running_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                val_total_preds += labels.size(0)
                val_correct_preds += (predicted == labels).sum().item()

                val_running_loss += loss.item()

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

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total_preds += labels.size(0)
            test_correct_preds += (predicted == labels).sum().item()

    # 计算测试集准确率
    test_accuracy = test_correct_preds / test_total_preds
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'sound_model_1.pth')

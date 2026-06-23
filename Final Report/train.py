import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from qbot_dataset import QbotLineFollowingDataset
import time


# ================= 1. 模型架构定义 =================
class QbotLineFollowerCNN(nn.Module):
    def __init__(self):
        super(QbotLineFollowerCNN, self).__init__()
        # 卷积层提取图像特征
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)

        self.flatten_size = 64 * 4 * 18

        # 全连接层输出回归预测
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        import torch.nn.functional as F
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ================= 2. 训练超参数设置 =================

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20


def train_model():
    print("正在加载数据集...")
    dataset = QbotLineFollowingDataset(data_dir="collected_data")
    # 设置 batch size 并打乱数据以提高模型泛化能力
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 实例化模型
    model = QbotLineFollowerCNN()

    # 将模型移动到 GPU (如果可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用设备进行训练: {device}")

    # ================= 3. 核心组件选择 =================
    # 损失函数 (Cost Function): 均方误差
    criterion = nn.MSELoss()

    # 优化器 (Optimizer): Adam
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ================= 4. 开始训练循环 =================
    print("\n开始训练模型...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()  # 设置为训练模式
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 打印每个 Epoch 的平均损失
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - 损失 (Loss): {epoch_loss:.6f}")

    end_time = time.time()
    print(f"\n训练完成！总耗时: {(end_time - start_time):.2f} 秒")

    # ================= 5. 保存模型 =================
    model_path = "qbot_cnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已成功保存至: {model_path}")


if __name__ == "__main__":
    train_model()
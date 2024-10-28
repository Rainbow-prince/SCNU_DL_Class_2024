import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout(0.3)

        # 3136 == 64 * 7 * 7 == 1/2 * 1/2 * (64 * 28 * 28)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一层卷积 relu激活 池化
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # 第二层卷积 relu激活 池化
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # 展平特征图
        x = torch.flatten(x, 1)

        # dropout
        x = self.dropout1(x)

        # 全连接层
        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        # 确保输出形状为 [Batch_sieze, 10]
        # print(out.size())
        assert out.ndim == 2 and out.size(0) == x.size(0) and out.size(1) == 10

        return out
